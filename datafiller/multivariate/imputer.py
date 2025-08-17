"""Core implementation of the DataFiller imputer."""

from typing import Iterable, Union

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from tqdm.auto import tqdm

from .._optimask import optimask
from ._numba_utils import (
    _imputable_rows,
    _index_to_mask,
    _subset,
    _subset_one_column,
    _trainable_rows,
    nan_positions,
    nan_positions_subset,
    unique2d,
    _mask_index_to_impute,
)
from ._utils import (
    _dataframe_cols_to_impute_to_indices,
    _dataframe_rows_to_impute_to_indices,
    _process_to_impute,
    _validate_input,
    scoring,
)


class MultivariateImputer:
    """Imputes missing values in a 2D array or pandas DataFrame.

    This class uses a model-based approach to fill in missing values, where
    each feature with missing values is predicted using other features in the
    dataset. It is designed to be efficient, using Numba for critical parts
    and finding optimal data subsets for model training.

    When the input is a pandas DataFrame, the imputer can handle categorical
    variables by using a classifier to impute them. Categorical features used
    for prediction are one-hot encoded.

    Args:
        regressor (RegressorMixin, optional): A scikit-learn compatible
            regressor for numerical features. Defaults to `Ridge()`.
        classifier (ClassifierMixin, optional): A scikit-learn compatible
            classifier for categorical features. Defaults to
            `OneVsRestClassifier(LogisticRegression())`.
        verbose (int, optional): The verbosity level. Defaults to 0.
        min_samples_train (int, optional): The minimum number of samples
            required to train a model. Defaults to 50.
        rng (int, optional): A seed for the random number generator. This is
            used for reproducible feature sampling when `n_nearest_features`
            is not None. Defaults to None.

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
        regressor: RegressorMixin = None,
        classifier: ClassifierMixin = None,
        verbose: int = 0,
        min_samples_train: int = 50,
        rng: Union[int, None] = None,
    ):
        self.regressor = Ridge() if regressor is None else regressor
        self.classifier = OneVsRestClassifier(LogisticRegression()) if classifier is None else classifier
        self.verbose = int(verbose)
        self.min_samples_train = min_samples_train
        self._rng = np.random.RandomState(rng)

    def _get_sampled_cols(
        self,
        n_features: int,
        n_nearest_features: int | None,
        scores: np.ndarray | None,
        scores_index: int,
    ) -> np.ndarray:
        """Selects the feature columns to use for imputing a specific column.

        If `n_nearest_features` is specified, it selects a subset of features
        based on the provided scores. Otherwise, it returns all features.

        Args:
            n_features: The total number of features.
            n_nearest_features: The number of features to select.
            scores: A matrix of scores for feature selection.
            scores_index: The index of the column being imputed in the
                scores matrix.

        Returns:
            An array of column indices to use for imputation.
        """
        if n_nearest_features is not None:
            p = scores[scores_index] / scores[scores_index].sum()
            p[np.isnan(p)] = 0
            if p.sum() == 0:
                p = None
            sampled_cols = self._rng.choice(
                a=np.arange(n_features),
                size=n_nearest_features,
                replace=False,
                p=p,
            )
            return np.sort(sampled_cols)
        return np.arange(n_features)

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
        is_categorical: bool = False,
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
        """
        m, n = x.shape

        sampled_cols = self._get_sampled_cols(n, n_nearest_features, scores, scores_index)

        # Exclude the column to impute from the features
        sampled_cols = np.setdiff1d(sampled_cols, [col_to_impute]).astype(np.uint32)

        imputable_rows = _imputable_rows(mask_nan=mask_nan, col=col_to_impute, mask_rows_to_impute=mask_rows_to_impute)
        if not len(imputable_rows):
            return

        trainable_rows = _trainable_rows(mask_nan=mask_nan, col=col_to_impute)
        if not len(trainable_rows):
            return  # Cannot impute if no training data is available for this column

        mask_trainable_rows = _index_to_mask(trainable_rows, m)
        mask_valid = ~mask_nan

        # Determine feature types for ColumnTransformer
        feature_types = [
            "categorical"
            if pd.api.types.is_object_dtype(x[:, c]) or pd.api.types.is_categorical_dtype(x[:, c])
            else "numerical"
            for c in sampled_cols
        ]

        categorical_features = sampled_cols[np.array(feature_types) == "categorical"]
        numerical_features = sampled_cols[np.array(feature_types) == "numerical"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ],
            remainder="drop",
        )

        X_transformed = preprocessor.fit_transform(x)

        # Impute NaNs in X_transformed with mean
        if np.isnan(X_transformed).any():
            col_means = np.nanmean(X_transformed, axis=0)
            nan_mask = np.isnan(X_transformed)
            X_transformed[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        patterns, indexes = unique2d(mask_valid[imputable_rows][:, sampled_cols])

        for k in range(len(patterns)):
            index_predict = imputable_rows[indexes == k]

            rows, _ = optimask(
                iy=iy,
                ix=ix,
                rows=trainable_rows,
                cols=sampled_cols,
                global_matrix_size=(m, n),
            )

            if len(rows) < self.min_samples_train:
                continue

            X_train = X_transformed[rows]
            y_train = x[rows, col_to_impute]

            estimator = self.classifier if is_categorical else self.regressor

            if is_categorical:
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)

            estimator.fit(X_train, y_train)

            X_predict = X_transformed[index_predict]
            predictions = estimator.predict(X_predict)

            if is_categorical:
                predictions = le.inverse_transform(predictions)

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
        if not isinstance(x, (np.ndarray, pd.DataFrame)):
            raise ValueError("x must be a numpy array or a pandas DataFrame.")

        if isinstance(x, np.ndarray) and x.ndim != 2:
            raise ValueError(f"x must be a 2D array, but got {x.ndim} dimensions.")

        is_df = isinstance(x, pd.DataFrame)
        if is_df:
            original_index = x.index
            original_columns = x.columns

            categorical_cols = x.select_dtypes(include=["category", "object"]).columns.tolist()
            numerical_cols = x.select_dtypes(include=np.number).columns.tolist()

            # Keep track of original dtypes
            original_dtypes = x.dtypes

            # Convert to numpy array
            x = x.to_numpy(dtype=object)
        else:
            categorical_cols = []
            numerical_cols = list(range(x.shape[1]))

        if is_df:
            cols_to_impute_indices = _dataframe_cols_to_impute_to_indices(cols_to_impute, original_columns)
        else:
            cols_to_impute_indices = cols_to_impute
        if is_df:
            rows_to_impute_indices = _dataframe_rows_to_impute_to_indices(rows_to_impute, original_index)
        else:
            rows_to_impute_indices = rows_to_impute
        n_nearest_features = _validate_input(x, rows_to_impute_indices, cols_to_impute_indices, n_nearest_features)

        m, n = x.shape
        rows_to_impute = _process_to_impute(size=m, to_impute=rows_to_impute_indices)

        if is_df:
            cols_to_impute = _dataframe_cols_to_impute_to_indices(cols_to_impute, original_columns)
        else:
            cols_to_impute = _process_to_impute(size=n, to_impute=cols_to_impute)

        mask_rows_to_impute = _mask_index_to_impute(size=m, to_impute=rows_to_impute)

        if n_nearest_features is not None:
            # Create a float version of x for scoring, replacing non-numeric with NaN
            x_float = np.full(x.shape, np.nan, dtype=float)
            for c in range(x.shape[1]):
                x_float[:, c] = pd.to_numeric(x[:, c], errors="coerce")
            scores = scoring(x_float, cols_to_impute)
        else:
            scores = None

        x_imputed = x.copy()

        # Create a mask for missing values, works with objects (e.g. pd.NA)
        mask_nan = pd.isna(x)
        iy, ix = np.where(mask_nan)

        for i, col_idx in enumerate(tqdm(cols_to_impute, leave=False, disable=(not self.verbose))):
            is_categorical = col_idx in [original_columns.get_loc(c) for c in categorical_cols] if is_df else False

            self._impute_col(
                x,
                x_imputed,
                col_idx,
                mask_nan,
                mask_rows_to_impute,
                iy,
                ix,
                n_nearest_features,
                scores,
                i,
                is_categorical,
            )

        if is_df:
            df_imputed = pd.DataFrame(x_imputed, index=original_index, columns=original_columns)
            # Restore original dtypes
            for col, dtype in original_dtypes.items():
                if df_imputed[col].dtype != dtype:
                    df_imputed[col] = df_imputed[col].astype(dtype)
            return df_imputed

        return x_imputed
