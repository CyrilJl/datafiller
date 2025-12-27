"""Core implementation of the DataFiller imputer."""

from typing import Iterable, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype, is_object_dtype, is_string_dtype
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm

from .._optimask import optimask
from ..estimators.ridge import FastRidge
from ._numba_utils import (
    _imputable_rows,
    _index_to_mask,
    _mask_index_to_impute,
    _subset,
    _subset_one_column,
    _trainable_rows,
    nan_positions,
    nan_positions_subset,
    unique2d,
)
from ._scoring import scoring
from ._utils import (
    _dataframe_cols_to_impute_to_indices,
    _dataframe_rows_to_impute_to_indices,
    _process_to_impute,
    _validate_input,
)


class MultivariateImputer:
    """Imputes missing values in a 2D numpy array.

    This class uses a model-based approach to fill in missing values, where
    each feature with missing values is predicted using other features in the
    dataset. It is designed to be efficient, using Numba for critical parts
    and finding optimal data subsets for model training. When a pandas
    DataFrame contains categorical, string, or boolean columns, they are
    one-hot encoded internally and imputed with a classifier before returning
    the original column layout.

    Args:
        regressor (RegressorMixin, optional): A scikit-learn compatible
            regressor. It should be a lightweight model, as it is fitted many
            times. By default, a custom Ridge implementation is used.
        classifier (ClassifierMixin, optional): A scikit-learn compatible
            classifier used for categorical and string targets. Defaults to
            ``DecisionTreeClassifier(max_depth=4)``.
        verbose (int, optional): The verbosity level. Defaults to 0.
        min_samples_train (int, optional): The minimum number of samples
            required to train a model. If, after the imputation, some values
            are still missing, it is likely that no training set with at least
            `min_samples_train` samples could be found. Defaults to `None`,
            which means that a model will be trained if at least one sample
            is available.
        rng (int, optional): A seed for the random number generator. This is
            used for reproducible feature sampling when `n_nearest_features`
            is not None. Defaults to None.
        scoring (str or callable, optional): The scoring function to use for
            feature selection.
            If 'default', the default scoring function is used.
            If a callable, it must take two arguments as input: the data matrix
            `X` (np.ndarray of shape `(n_samples, n_features)`) and the
            columns to impute `cols_to_impute` (np.ndarray of shape
            `(n_cols_to_impute,)`), and return a score matrix of shape
            `(n_cols_to_impute, n_features)`.
            Defaults to 'default'.

    Attributes:
        imputation_features_ (dict or None): A dictionary mapping each imputed
            column to the features used to impute it. This attribute is only
            populated when `n_nearest_features` is not None. If the input is a
            pandas DataFrame, the keys and values will be column names. If the
            input is a NumPy array, they will be integer indices.

    Examples:
        .. code-block:: python

            import numpy as np
            from datafiller import MultivariateImputer

            # Create a matrix with missing values
            X = np.array([
                [1, 2, 3],
                [4, np.nan, 6],
                [7, 8, 9]
            ])

            # Create an imputer and fill the missing values
            imputer = MultivariateImputer()
            X_imputed = imputer(X)

            print(X_imputed)
    """

    def __init__(
        self,
        *,
        regressor: RegressorMixin | None = None,
        classifier: ClassifierMixin | None = None,
        verbose: int = 0,
        min_samples_train: int | None = None,
        rng: Union[int, None] = None,
        scoring: Union[str, callable] = "default",
    ):
        """
        Args:
            regressor: Regressor used to impute numerical targets. Defaults to ``FastRidge``.
            classifier: Classifier used to impute categorical or string targets.
                Defaults to ``DecisionTreeClassifier(max_depth=4)``.
        """
        self.regressor = regressor or FastRidge()
        self.classifier = classifier or DecisionTreeClassifier(max_depth=4)
        self.verbose = int(verbose)
        if min_samples_train is None:
            self.min_samples_train = 1
        else:
            self.min_samples_train = min_samples_train
        self._rng = np.random.RandomState(rng)
        if scoring == "default":
            self.scoring = scoring
        elif callable(scoring):
            self.scoring = scoring
        else:
            raise ValueError("`scoring` must be 'default' or a callable.")
        self.imputation_features_ = None

    @np.errstate(all="ignore")
    def _get_sampled_cols(
        self,
        n_features: int,
        col_to_impute: int,
        n_nearest_features: int | None,
        scores: np.ndarray | None,
        scores_index: int,
    ) -> np.ndarray:
        """Selects the feature columns to use for imputing a specific column.
        If `n_nearest_features` is specified, it selects a subset of features
        based on the provided scores. Otherwise, it returns all features.
        Args:
            n_features: The total number of features.
            col_to_impute: The index of the column to impute.
            n_nearest_features: The number of features to select.
            scores: A matrix of scores for feature selection.
            scores_index: The index of the column being imputed in the
                scores matrix.
        Returns:
            An array of column indices to use for imputation.
        """
        cols_to_sample_from = np.arange(n_features)
        cols_to_sample_from = cols_to_sample_from[cols_to_sample_from != col_to_impute]

        if n_nearest_features is not None:
            # The scores are for all n_features, but we are sampling from n_features - 1
            # The scores array is (n_cols_to_impute, n_features)
            # The scores for the column to impute against itself should be 0 or NaN.
            p = scores[scores_index][cols_to_sample_from]
            p = p / p.sum()
            p[np.isnan(p)] = 0
            if p.sum() == 0:
                p = None
            n_nearest_features = min(n_nearest_features, len(cols_to_sample_from))
            sampled_cols = self._rng.choice(
                a=cols_to_sample_from,
                size=n_nearest_features,
                replace=False,
                p=p,
            )
            return np.sort(sampled_cols)
        return cols_to_sample_from

    def _encode_dataframe(self, df: pd.DataFrame) -> dict:
        """Encode a pandas DataFrame into a numeric matrix suitable for imputation."""
        encoded_arrays = []
        encoded_feature_names: list[str] = []
        main_column_indices: list[int] = []
        categorical_targets: dict[int, list] = {}
        encoded_index_to_original: dict[int, str] = {}
        original_dtypes = df.dtypes.to_dict()

        for col in df.columns:
            series = df[col]
            is_categorical = any(
                [
                    isinstance(series.dtype, pd.CategoricalDtype),
                    is_object_dtype(series.dtype),
                    is_string_dtype(series.dtype),
                    is_bool_dtype(series.dtype),
                ]
            )

            main_idx = len(encoded_feature_names)
            encoded_index_to_original[main_idx] = col
            main_column_indices.append(main_idx)
            encoded_feature_names.append(col)

            if is_categorical:
                if isinstance(series.dtype, pd.CategoricalDtype):
                    categories = series.cat.categories.tolist()
                else:
                    categories = pd.Categorical(series.dropna()).categories.tolist()
                cat_series = pd.Categorical(series, categories=categories)
                codes = cat_series.codes.astype(np.float32)
                codes[codes == -1] = np.nan
                categorical_targets[main_idx] = categories
                encoded_arrays.append(codes.reshape(-1, 1))

                dummy_df = pd.get_dummies(series, prefix=col, dummy_na=False)
                if len(dummy_df.columns):
                    if series.isna().any():
                        dummy_df = dummy_df.mask(series.isna())
                    dummy_df = dummy_df.astype(np.float32)
                    encoded_feature_names.extend(dummy_df.columns.tolist())
                    encoded_arrays.append(dummy_df.to_numpy(dtype=np.float32, copy=False))
            else:
                encoded_arrays.append(series.to_numpy(dtype=np.float32).reshape(-1, 1))

        encoded_matrix = np.concatenate(encoded_arrays, axis=1).astype(np.float32, copy=False)
        return {
            "data": encoded_matrix,
            "main_column_indices": np.array(main_column_indices, dtype=int),
            "encoded_feature_names": encoded_feature_names,
            "categorical_targets": categorical_targets,
            "encoded_index_to_original": encoded_index_to_original,
            "original_dtypes": original_dtypes,
        }

    def _cast_series_to_dtype(self, series: pd.Series, dtype) -> pd.Series:
        """Cast a numeric series back to the original dtype."""
        if is_integer_dtype(dtype):
            rounded = series.round()
            try:
                return rounded.astype(dtype)
            except (TypeError, ValueError):
                return rounded.astype(pd.Int64Dtype())
        if is_float_dtype(dtype):
            return series.astype(dtype)
        return series.astype(dtype)

    def _decode_dataframe(
        self,
        x_imputed: np.ndarray,
        original_index: pd.Index,
        original_columns: pd.Index,
        main_column_indices: np.ndarray,
        categorical_targets: dict[int, list],
        original_dtypes: dict,
    ) -> pd.DataFrame:
        """Decode an imputed numeric matrix back to the original DataFrame layout."""
        data = {}
        for i, col in enumerate(original_columns):
            encoded_idx = main_column_indices[i]
            col_data = x_imputed[:, encoded_idx]

            if encoded_idx in categorical_targets:
                categories = categorical_targets[encoded_idx]
                mask = np.isnan(col_data)
                decoded = np.full(len(col_data), np.nan, dtype=object)
                if len(categories) and np.any(~mask):
                    category_values = np.array(categories, dtype=object)
                    decoded[~mask] = category_values[col_data[~mask].astype(np.int64)]

                dtype = original_dtypes[col]
                if is_bool_dtype(dtype):
                    series = pd.Series(decoded, index=original_index, dtype="boolean")
                elif isinstance(dtype, pd.CategoricalDtype):
                    dtype_categories = getattr(dtype, "categories", None)
                    series = pd.Series(
                        pd.Categorical(
                            decoded,
                            categories=dtype_categories if dtype_categories is not None else categories,
                            ordered=getattr(dtype, "ordered", False),
                        ),
                        index=original_index,
                    )
                elif is_string_dtype(dtype):
                    series = pd.Series(decoded, index=original_index, dtype="string")
                else:
                    series = pd.Series(decoded, index=original_index)
            else:
                series = pd.Series(col_data, index=original_index)
                series = self._cast_series_to_dtype(series, original_dtypes[col])

            data[col] = series

        return pd.DataFrame(data, index=original_index, columns=original_columns)

    def _impute_col(
        self,
        x: np.ndarray,
        x_imputed: np.ndarray,
        col_to_impute: int,
        mask_nan: np.ndarray,
        mask_rows_to_impute: np.ndarray,
        iy: np.ndarray,
        ix: np.ndarray,
        n_nearest_features: int | None,
        scores: np.ndarray | None,
        scores_index: int,
        categorical_cols: set[int],
    ) -> None:
        """Imputes all missing values in a single column.

        It identifies patterns of missingness, finds optimal data subsets for
        training, fits the estimator, and predicts the missing values.

        Args:
            x (np.ndarray): The original data matrix.
            x_imputed (np.ndarray): The matrix where imputed values are stored.
            col_to_impute (int): The index of the column to impute.
            mask_nan (np.ndarray): A boolean mask of NaNs for the entire matrix.
            mask_rows_to_impute (np.ndarray): A boolean mask of rows to be imputed.
            iy (np.ndarray): Row indices of all NaNs.
            ix (np.ndarray): Column indices of all NaNs.
            n_nearest_features (int | None): The number of features to use.
            scores (np.ndarray | None): The feature selection scores.
            scores_index (int): The index of the column being imputed in the
                scores matrix.
            categorical_cols (set[int]): Indices of columns that should be
                treated as categorical targets.
        """
        m, n = x.shape

        imputable_rows = _imputable_rows(mask_nan=mask_nan, col=col_to_impute, mask_rows_to_impute=mask_rows_to_impute)
        if not len(imputable_rows):
            return

        sampled_cols = self._get_sampled_cols(n, col_to_impute, n_nearest_features, scores, scores_index)

        if self.imputation_features_ is not None:
            self.imputation_features_[col_to_impute] = sampled_cols

        trainable_rows = _trainable_rows(mask_nan=mask_nan, col=col_to_impute)
        if not len(trainable_rows):
            return  # Cannot impute if no training data is available for this column

        mask_trainable_rows = _index_to_mask(trainable_rows, m)
        mask_valid = ~mask_nan
        patterns, indexes = unique2d(mask_valid[imputable_rows][:, sampled_cols])

        pre_iy_trial, pre_ix_trial = nan_positions_subset(
            iy,
            ix,
            mask_trainable_rows,
            _index_to_mask(sampled_cols, n),
        )

        for k in range(len(patterns)):
            index_predict = imputable_rows[indexes == k]
            usable_cols = sampled_cols[patterns[k]].astype(np.uint32)
            mask_usable_cols = _index_to_mask(usable_cols, n)
            if len(usable_cols):
                iy_trial, ix_trial = nan_positions_subset(
                    pre_iy_trial,
                    pre_ix_trial,
                    mask_trainable_rows,
                    mask_usable_cols,
                )
                rows, cols = optimask(
                    iy=iy_trial,
                    ix=ix_trial,
                    rows=trainable_rows,
                    cols=usable_cols,
                    global_matrix_size=(m, n),
                )
                if (len(rows) < self.min_samples_train) or (not len(cols)):
                    continue  # Not enough data to train a model

                X_train = _subset(X=x, rows=rows, columns=cols)
                y_train = _subset_one_column(X=x, rows=rows, col=col_to_impute)
                is_categorical_target = col_to_impute in categorical_cols
                if is_categorical_target:
                    unique_y = np.unique(y_train)
                    if len(unique_y) < 2:
                        x_imputed[index_predict, col_to_impute] = unique_y[0]
                        continue
                    estimator = self.classifier
                    y_train = y_train.astype(np.int64)
                else:
                    estimator = self.regressor
                estimator.fit(X=X_train, y=y_train)
                predictions = estimator.predict(_subset(X=x, rows=index_predict, columns=cols))
                if is_categorical_target:
                    predictions = predictions.astype(np.float32)
                x_imputed[index_predict, col_to_impute] = predictions

    def __call__(
        self,
        x: Union[np.ndarray, pd.DataFrame],
        rows_to_impute: None | int | Iterable[int] | Iterable[str] = None,
        cols_to_impute: None | int | Iterable[int] | Iterable[str] = None,
        n_nearest_features: None | float | int = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Imputes missing values in the input data.

        The method can handle both NumPy arrays and pandas DataFrames.

        Args:
            x: The input data matrix with missing values (NaNs).
                Can be a numpy array or a pandas DataFrame.
            rows_to_impute: The rows to impute. The interpretation of this
                argument depends on the type of `x`.
                - If `x` is a NumPy array, this must be a list of integer indices.
                - If `x` is a pandas DataFrame, this must be a list of index labels.
                If None, all rows are considered for imputation. Defaults to None.
            cols_to_impute: The columns to impute. The interpretation of this
                argument depends on the type of `x`.
                - If `x` is a NumPy array, this must be a list of integer indices.
                - If `x` is a pandas DataFrame, this must be a list of column labels.
                If None, all columns are considered for imputation. Defaults to None.
            n_nearest_features: The number of features to use for
                imputation. If it's an int, it's the absolute number of
                features. If it's a float, it's the fraction of features to
                use. If None, all features are used. Defaults to None.

        Returns:
            The imputed data matrix. The return type will match the input type
            (NumPy array or pandas DataFrame).
        """
        is_df = isinstance(x, pd.DataFrame)
        categorical_targets: dict[int, list] = {}
        encoded_feature_names: list[str] | None = None
        encoded_index_to_original: dict[int, str] = {}
        original_index = None
        original_columns = None
        main_column_indices = None
        original_dtypes = None

        if is_df:
            original_index = x.index
            original_columns = x.columns
            rows_to_impute = _dataframe_rows_to_impute_to_indices(rows_to_impute, original_index)
            cols_to_impute_df = _dataframe_cols_to_impute_to_indices(cols_to_impute, original_columns)
            cols_to_impute_processed = _process_to_impute(size=len(original_columns), to_impute=cols_to_impute_df)

            encoded = self._encode_dataframe(x)
            x = encoded["data"]
            main_column_indices = encoded["main_column_indices"]
            categorical_targets = encoded["categorical_targets"]
            encoded_feature_names = encoded["encoded_feature_names"]
            encoded_index_to_original = encoded["encoded_index_to_original"]
            original_dtypes = encoded["original_dtypes"]
            cols_to_impute = np.array([main_column_indices[idx] for idx in cols_to_impute_processed], dtype=np.int64)
        else:
            x = np.asarray(x)

        n_nearest_features = _validate_input(x, rows_to_impute, cols_to_impute, n_nearest_features)

        m, n = x.shape
        rows_to_impute = _process_to_impute(size=m, to_impute=rows_to_impute)
        cols_to_impute = _process_to_impute(size=n, to_impute=cols_to_impute)
        mask_rows_to_impute = _mask_index_to_impute(size=m, to_impute=rows_to_impute)
        categorical_cols = set(categorical_targets.keys())

        if n_nearest_features is not None:
            if self.scoring == "default":
                scores = scoring(x, cols_to_impute)
            else:
                scores = self.scoring(x, cols_to_impute)
            self.imputation_features_ = {}
        else:
            scores = None
            self.imputation_features_ = None

        x_imputed = x.copy()
        mask_nan, iy, ix = nan_positions(x)

        for i, col in enumerate(tqdm(cols_to_impute, leave=False, disable=(not self.verbose))):
            self._impute_col(
                x,
                x_imputed,
                col,
                mask_nan,
                mask_rows_to_impute,
                iy,
                ix,
                n_nearest_features,
                scores,
                i,
                categorical_cols,
            )

        if is_df and self.imputation_features_ is not None:
            assert encoded_feature_names is not None
            self.imputation_features_ = {
                encoded_index_to_original.get(col, encoded_feature_names[col]): [
                    encoded_index_to_original.get(feature, encoded_feature_names[feature]) for feature in features
                ]
                for col, features in self.imputation_features_.items()
            }

        if is_df:
            return self._decode_dataframe(
                x_imputed=x_imputed,
                original_index=original_index,
                original_columns=original_columns,
                main_column_indices=main_column_indices,
                categorical_targets=categorical_targets,
                original_dtypes=original_dtypes,
            )

        return x_imputed
