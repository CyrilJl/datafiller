"""Core implementation of the DataFiller imputer."""

import warnings
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype, is_object_dtype, is_string_dtype
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm

from .._optimask import optimask
from ..estimators.ridge import FastRidge, fit_ridge_from_gram
from ..exceptions import DataFillerTypeError, DataFillerValueError
from ._gpu import GramBackend
from ._numba_utils import (
    _imputable_rows,
    _index_to_mask,
    _mask_index_to_impute,
    _subset,
    _subset_one_column,
    _trainable_rows,
    complete_rows_excluding,
    extra_rows_excluding,
    nan_cols_csc,
    nan_positions,
    nan_positions_subset_cols,
    unique2d,
)
from ._polars import (
    decode_polars_dataframe,
    encode_polars_dataframe,
    is_polars_dataframe,
    is_polars_lazyframe,
    polars_cols_to_indices,
    validate_polars_rows,
)
from ._scoring import scoring
from ._utils import (
    _dataframe_cols_to_impute_to_indices,
    _dataframe_rows_to_impute_to_indices,
    _process_to_impute,
    _validate_input,
)


class MultivariateImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values in a 2D numpy array.

    This class uses a model-based approach to fill in missing values, where
    each feature with missing values is predicted using other features in the
    dataset. It is designed to be efficient, using Numba for critical parts
    and finding optimal data subsets for model training. When a pandas or
    Polars DataFrame contains categorical, string, or boolean columns, they are
    one-hot encoded internally and imputed with a classifier before returning
    the original column layout.

    Rows to impute are grouped by their pattern of observed features, and the
    training data for each pattern is selected along a fixed three-step path:

    1. **Complete rows** — the rows fully observed on the pattern's features
       are used directly when at least `min_samples_train` of them exist.
    2. **optimask** — otherwise, the `optimask` algorithm searches for the
       largest NaN-free rectangular subset, trading feature columns for
       training rows and preferring rectangles that keep at least
       `min_samples_train` rows.
    3. **Fallback** — cells whose pattern still cannot reach the threshold
       are filled by the `fallback` strategy (column mean / most frequent
       category by default, or left NaN with ``fallback=None``).

    Args:
        regressor (RegressorMixin, optional): A scikit-learn compatible
            regressor. It should be a lightweight model, as it is fitted many
            times. By default, a custom Ridge implementation is used.
        classifier (ClassifierMixin, optional): A scikit-learn compatible
            classifier used for categorical and string targets. Defaults to
            ``DecisionTreeClassifier(max_depth=4, random_state=rng)``.
        verbose (int, optional): The verbosity level. Defaults to 0.
        min_samples_train (int, optional): The minimum number of samples
            required to train a model. Patterns with fewer complete rows fall
            back to an `optimask` search that trades feature columns for
            training rows; cells whose patterns still cannot reach the
            threshold are handled by `fallback`. Defaults to `None`, which
            resolves to 20 — calibrated on real and synthetic datasets, where
            values of 10-20 consistently beat permissive ones (fits on fewer
            samples are often worse than a plain column mean once missingness
            reaches ~25%).
        fallback (str or None, optional): What to do with cells no model
            could impute (their pattern never reached `min_samples_train`
            training rows). ``"simple"`` (default) fills them with the column
            mean (most frequent category for categorical columns). ``None``
            leaves them as NaN.
        rng (int, optional): A seed for the random number generator. This is
            used for reproducible feature sampling when `n_nearest_features`
            is not None, and for the default categorical classifier when one
            is not provided. Defaults to None.
        scoring (str or callable, optional): The scoring function to use for
            feature selection.
            If 'default', the default scoring function is used.
            If a callable, it must take two arguments as input: the data matrix
            `X` (np.ndarray of shape `(n_samples, n_features)`) and the
            columns to impute `cols_to_impute` (np.ndarray of shape
            `(n_cols_to_impute,)`), and return a score matrix of shape
            `(n_cols_to_impute, n_features)`.
            Defaults to 'default'.
        device (str, optional): Device used to solve the default ridge
            models, e.g. ``"cuda"`` or ``"cuda:0"``. Requires the optional
            PyTorch dependency (``pip install datafiller[gpu]``). All
            missingness patterns of a column are then solved as batched
            tensor operations, which is considerably faster when many
            columns are imputed on large matrices. Categorical targets and
            patterns with fewer than `min_samples_train` complete rows
            still use the CPU implementation, and a custom regressor
            ignores `device` entirely (a UserWarning is emitted). If None
            (default), the pure NumPy/Numba CPU path is used and PyTorch
            is never imported.

    Attributes:
        imputation_features_ (dict or None): A dictionary mapping each imputed
            column to the features used to impute it. This attribute is only
            populated when `n_nearest_features` is not None. If the input is a
            pandas or Polars DataFrame, the keys and values will be column names. If the
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

    _DEFAULT_MIN_SAMPLES_TRAIN = 20

    @classmethod
    def _resolve_min_samples_train(cls, min_samples_train: int | None) -> int:
        return cls._DEFAULT_MIN_SAMPLES_TRAIN if min_samples_train is None else min_samples_train

    @staticmethod
    def _validate_fallback(fallback: str | None) -> str | None:
        if fallback not in (None, "simple"):
            raise DataFillerValueError(f"fallback must be 'simple' or None, got {fallback!r}")
        return fallback

    def __init__(
        self,
        *,
        regressor: RegressorMixin | None = None,
        classifier: ClassifierMixin | None = None,
        verbose: int = 0,
        min_samples_train: int | None = None,
        fallback: str | None = "simple",
        rng: int | None = None,
        scoring: str | Callable = "default",
        device: str | None = None,
    ):
        """
        Args:
            regressor: Regressor used to impute numerical targets. Defaults to ``FastRidge``.
            classifier: Classifier used to impute categorical or string targets.
                Defaults to ``DecisionTreeClassifier(max_depth=4, random_state=rng)``.
            fallback: ``"simple"`` fills cells no model could impute with the column
                mean (mode for categoricals); ``None`` leaves them as NaN.
            device: Optional torch device (e.g. ``"cuda"``) for batched ridge solves.
        """
        self._regressor_default = regressor is None
        # Regressors and classifiers are sklearn-compatible duck types; the
        # mixin base classes do not declare fit/predict, hence Any.
        self.regressor: Any = regressor or FastRidge()
        self.verbose = int(verbose)
        self.min_samples_train = self._resolve_min_samples_train(min_samples_train)
        self.fallback = self._validate_fallback(fallback)
        self.rng = rng
        self._rng = np.random.RandomState(rng)
        self._classifier_default = classifier is None
        self.classifier: Any = classifier or DecisionTreeClassifier(max_depth=4, random_state=rng)
        if scoring != "default" and not callable(scoring):
            raise DataFillerValueError("`scoring` must be 'default' or a callable.")
        self.scoring: str | Callable[..., np.ndarray] = scoring
        self.device = device
        self._gpu_backend: GramBackend | None = None
        self.imputation_features_: dict | None = None

    def fit(self, X, y: None = None) -> "MultivariateImputer":
        """No-op fit for sklearn compatibility."""
        return self

    def transform(self, X):
        """Impute missing values in X using stored configuration."""
        return self(X)

    def set_params(self, **params) -> "MultivariateImputer":
        """Set parameters and refresh derived attributes."""
        params = params.copy()
        if "min_samples_train" in params:
            params["min_samples_train"] = self._resolve_min_samples_train(params["min_samples_train"])
        if "fallback" in params:
            params["fallback"] = self._validate_fallback(params["fallback"])

        classifier_param = params.get("classifier", None) if "classifier" in params else None
        regressor_param = params.get("regressor", None) if "regressor" in params else None
        rng_changed = "rng" in params

        super().set_params(**params)

        if "classifier" in params:
            self._classifier_default = classifier_param is None
            if self._classifier_default:
                self.classifier = DecisionTreeClassifier(max_depth=4, random_state=self.rng)

        if "regressor" in params:
            self._regressor_default = regressor_param is None
            if self._regressor_default:
                self.regressor = FastRidge()

        if rng_changed:
            self._rng = np.random.RandomState(self.rng)
            if self._classifier_default:
                self.classifier = DecisionTreeClassifier(max_depth=4, random_state=self.rng)

        return self

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
            assert scores is not None
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

    @staticmethod
    @np.errstate(all="ignore")
    def _masked_column_stats(
        x: np.ndarray,
        mask_nan: np.ndarray,
        cols: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Column means and standard deviations over observed values.

        Equivalent to ``np.nanmean``/``np.nanstd`` on ``x[:, cols]`` but with a
        single full-size temporary instead of several, using a numerically
        stable two-pass computation. All-NaN columns get mean 0, and zero or
        undefined scales are replaced by 1 so normalization is a no-op there.
        """
        if len(cols) == x.shape[1]:
            sub_x, sub_mask = x, mask_nan
        else:
            sub_x, sub_mask = x[:, cols], mask_nan[:, cols]

        work_dtype = sub_x.dtype if sub_x.dtype == np.float32 else np.float64
        counts = (len(sub_x) - sub_mask.sum(axis=0)).astype(work_dtype)
        z = np.where(sub_mask, 0, sub_x).astype(work_dtype, copy=False)
        means = z.sum(axis=0) / counts

        np.subtract(sub_x, means, out=z, casting="unsafe")
        z[sub_mask] = 0
        scales = np.sqrt(np.einsum("ij,ij->j", z, z) / counts)

        means = np.where(np.isnan(means), 0.0, means)
        scales = np.where((scales == 0) | np.isnan(scales), 1.0, scales)
        return means, scales

    @staticmethod
    def _group_pattern_rows(indexes: np.ndarray) -> list[np.ndarray]:
        """Group inverse-index labels once instead of rescanning them per pattern."""
        if not len(indexes):
            return []
        order = np.argsort(indexes, kind="stable").astype(np.uint32, copy=False)
        sorted_indexes = indexes[order]
        split_points = np.flatnonzero(np.diff(sorted_indexes)) + 1
        return [group.astype(np.uint32, copy=False) for group in np.split(order, split_points)]

    def _impute_col(
        self,
        x: np.ndarray,
        x_imputed: np.ndarray,
        col_to_impute: int,
        mask_nan: np.ndarray,
        mask_rows_to_impute: np.ndarray,
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
            n_nearest_features (int | None): The number of features to use.
            scores (np.ndarray | None): The feature selection scores.
            scores_index (int): The index of the column being imputed in the
                scores matrix.
            categorical_cols (set[int]): Indices of columns that should be
                treated as categorical targets.
        """
        _, n = x.shape

        if not (
            imputable_rows := _imputable_rows(
                mask_nan=mask_nan, col=col_to_impute, mask_rows_to_impute=mask_rows_to_impute
            )
        ).size:
            return

        sampled_cols = self._get_sampled_cols(n, col_to_impute, n_nearest_features, scores, scores_index)

        if self.imputation_features_ is not None:
            self.imputation_features_[col_to_impute] = sampled_cols

        if not (trainable_rows := _trainable_rows(mask_nan=mask_nan, col=col_to_impute)).size:
            return  # Cannot impute if no training data is available for this column

        is_categorical_target = col_to_impute in categorical_cols
        # For the default ridge regressor, models are solved from Gram matrices
        # accumulated over the rows complete on each pattern's usable columns,
        # instead of refitting on a materialized row subset for every
        # missingness pattern.
        use_gram = (not is_categorical_target) and type(self.regressor) is FastRidge

        if use_gram and self._gpu_backend is not None:
            return self._impute_col_gpu(x, x_imputed, col_to_impute, imputable_rows, trainable_rows, sampled_cols)

        sampled_cols_uint32 = sampled_cols.astype(np.uint32, copy=False)
        local_train = _subset(X=x, rows=trainable_rows, columns=sampled_cols_uint32)
        local_target = _subset_one_column(X=x, rows=trainable_rows, col=col_to_impute)
        local_predict = _subset(X=x, rows=imputable_rows, columns=sampled_cols_uint32)

        patterns, indexes = unique2d(~np.isnan(local_predict))
        prediction_groups = self._group_pattern_rows(indexes)

        local_mask_nan, local_iy, local_ix = nan_positions(local_train)
        m_local = len(trainable_rows)
        k_local = len(sampled_cols_uint32)
        local_rows = np.arange(m_local, dtype=np.uint32)
        local_cols = np.arange(k_local, dtype=np.uint32)

        row_nan_count = local_mask_nan.sum(axis=1).astype(np.uint32, copy=False)
        col_ptr, col_rows = nan_cols_csc(local_iy, local_ix, k_local)
        hits = np.zeros(m_local, dtype=np.uint32)
        stamp = np.full(m_local, -1, dtype=np.int64)
        epoch = np.int64(0)

        # The Gram of `[X, y, 1]` is accumulated once over the globally
        # complete rows; each pattern then only adds a small correction.
        if use_gram:
            z_aug = np.empty((m_local, k_local + 2), dtype=np.float32)
            z_aug[:, :k_local] = local_train
            z_aug[:, k_local] = local_target
            z_aug[:, k_local + 1] = 1.0
            complete0 = np.flatnonzero(row_nan_count == 0).astype(np.uint32, copy=False)
            z0 = z_aug if len(complete0) == m_local else z_aug[complete0]
            gram0 = z0.T @ z0

        training_groups: dict[tuple, dict[str, Any]] = {}
        for pattern, prediction_group in zip(patterns, prediction_groups, strict=False):
            usable_cols_local = local_cols[pattern].astype(np.uint32, copy=False)
            if not len(usable_cols_local):
                continue
            excluded_cols_local = local_cols[~pattern].astype(np.uint32, copy=False)
            epoch += 1

            if use_gram:
                extras, n_complete = extra_rows_excluding(
                    row_nan_count, col_ptr, col_rows, excluded_cols_local, hits, stamp, epoch
                )
                if n_complete >= self.min_samples_train:
                    key = (True, usable_cols_local.tobytes())
                    if key not in training_groups:
                        training_groups[key] = {
                            "rows": None,
                            "gram_extras": extras,
                            "n_samples": int(n_complete),
                            "cols": usable_cols_local,
                            "prediction_groups": [],
                        }
                    training_groups[key]["prediction_groups"].append(prediction_group)
                    continue
            else:
                rows = complete_rows_excluding(
                    row_nan_count, col_ptr, col_rows, excluded_cols_local, hits, stamp, epoch
                )
                if len(rows) >= self.min_samples_train:
                    key = (False, rows.tobytes(), usable_cols_local.tobytes())
                    if key not in training_groups:
                        training_groups[key] = {"rows": rows, "cols": usable_cols_local, "prediction_groups": []}
                    training_groups[key]["prediction_groups"].append(prediction_group)
                    continue

            mask_usable_cols = _index_to_mask(usable_cols_local, k_local)
            iy_trial, ix_trial = nan_positions_subset_cols(local_iy, local_ix, mask_usable_cols)
            rows, cols = optimask(
                iy=iy_trial,
                ix=ix_trial,
                rows=local_rows,
                cols=usable_cols_local,
                global_matrix_size=local_train.shape,
                copy=False,
                min_rows=self.min_samples_train,
            )
            if (len(rows) < self.min_samples_train) or (not len(cols)):
                continue  # Not enough data to train a model

            key = (False, rows.tobytes(), cols.tobytes())
            if key not in training_groups:
                training_groups[key] = {"rows": rows, "cols": cols, "prediction_groups": []}
            training_groups[key]["prediction_groups"].append(prediction_group)

        for group in training_groups.values():
            cols = group["cols"]
            predict_rows = (
                group["prediction_groups"][0]
                if len(group["prediction_groups"]) == 1
                else np.concatenate(group["prediction_groups"]).astype(np.uint32, copy=False)
            )

            if group.get("rows") is None:
                aug_idx = np.concatenate([cols, [k_local, k_local + 1]]).astype(np.uint32, copy=False)
                gram = gram0[np.ix_(aug_idx, aug_idx)]
                extras = group["gram_extras"]
                if extras.size:
                    z_extra = _subset(X=z_aug, rows=extras, columns=aug_idx)
                    gram = gram + z_extra.T @ z_extra
                coef, intercept = fit_ridge_from_gram(
                    gram=gram,
                    n_samples=group["n_samples"],
                    alpha=self.regressor.alpha,
                    fit_intercept=self.regressor.fit_intercept,
                )
                x_pred = _subset(X=local_predict, rows=predict_rows, columns=cols)
                predictions = x_pred.astype(np.float32, copy=False) @ coef + intercept
                x_imputed[imputable_rows[predict_rows], col_to_impute] = predictions
                continue

            rows = group["rows"]
            X_train = _subset(X=local_train, rows=rows, columns=cols)
            y_train = local_target[rows]

            if is_categorical_target:
                if (unique_y := np.unique(y_train)).size < 2:
                    x_imputed[imputable_rows[predict_rows], col_to_impute] = unique_y[0]
                    continue
                estimator = self.classifier
                y_train = y_train.astype(np.int64)
            else:
                estimator = self.regressor

            estimator.fit(X=X_train, y=y_train)
            predictions = estimator.predict(_subset(X=local_predict, rows=predict_rows, columns=cols))
            if is_categorical_target:
                predictions = predictions.astype(np.float32)
            x_imputed[imputable_rows[predict_rows], col_to_impute] = predictions

    def _impute_col_gpu(
        self,
        x: np.ndarray,
        x_imputed: np.ndarray,
        col_to_impute: int,
        imputable_rows: np.ndarray,
        trainable_rows: np.ndarray,
        sampled_cols: np.ndarray,
    ) -> None:
        """Device-batched variant of the default-ridge Gram fast path.

        All missingness patterns of the column are solved in one batched pass
        on ``self.device``; patterns with fewer than `min_samples_train`
        complete rows fall back to the CPU optimask branch below.
        """
        assert self._gpu_backend is not None
        result = self._gpu_backend.impute_column(
            x=x,
            col=col_to_impute,
            trainable_rows=trainable_rows,
            imputable_rows=imputable_rows,
            sampled_cols=sampled_cols,
            alpha=self.regressor.alpha,
            fit_intercept=self.regressor.fit_intercept,
            min_samples_train=self.min_samples_train,
        )
        if result.row_valid.any():
            rows_solved = np.flatnonzero(result.row_valid)
            x_imputed[imputable_rows[rows_solved], col_to_impute] = result.predictions[rows_solved]
        if result.all_valid:
            return
        assert result.patterns is not None and result.indexes is not None and result.pattern_valid is not None

        sampled_cols_uint32 = sampled_cols.astype(np.uint32, copy=False)
        local_train = _subset(X=x, rows=trainable_rows, columns=sampled_cols_uint32)
        local_target = _subset_one_column(X=x, rows=trainable_rows, col=col_to_impute)
        local_predict = _subset(X=x, rows=imputable_rows, columns=sampled_cols_uint32)
        prediction_groups = self._group_pattern_rows(result.indexes)
        _, local_iy, local_ix = nan_positions(local_train)
        m_local, k_local = local_train.shape
        local_rows = np.arange(m_local, dtype=np.uint32)
        local_cols = np.arange(k_local, dtype=np.uint32)

        for p in np.flatnonzero(~result.pattern_valid):
            pattern = result.patterns[p]
            prediction_group = prediction_groups[p]
            usable_cols_local = local_cols[pattern].astype(np.uint32, copy=False)
            if not len(usable_cols_local):
                continue
            mask_usable_cols = _index_to_mask(usable_cols_local, k_local)
            iy_trial, ix_trial = nan_positions_subset_cols(local_iy, local_ix, mask_usable_cols)
            rows, cols = optimask(
                iy=iy_trial,
                ix=ix_trial,
                rows=local_rows,
                cols=usable_cols_local,
                global_matrix_size=local_train.shape,
                copy=False,
                min_rows=self.min_samples_train,
            )
            if (len(rows) < self.min_samples_train) or (not len(cols)):
                continue  # Not enough data to train a model
            X_train = _subset(X=local_train, rows=rows, columns=cols)
            y_train = local_target[rows]
            self.regressor.fit(X=X_train, y=y_train)
            predictions = self.regressor.predict(_subset(X=local_predict, rows=prediction_group, columns=cols))
            x_imputed[imputable_rows[prediction_group], col_to_impute] = predictions

    @staticmethod
    def _apply_fallback(
        x_imputed: np.ndarray,
        mask_nan: np.ndarray,
        mask_rows_to_impute: np.ndarray,
        cols_to_impute: np.ndarray,
        categorical_cols: set,
    ) -> None:
        """Fills cells no model could impute with the column mean (mode for categoricals).

        Only cells that were targeted for imputation (NaN in the input, row
        selected) are filled; columns with no observed value at all are left
        untouched.
        """
        for col in cols_to_impute:
            remaining = mask_nan[:, col] & mask_rows_to_impute & np.isnan(x_imputed[:, col])
            if not remaining.any():
                continue
            observed = x_imputed[~mask_nan[:, col], col]
            if not observed.size:
                continue
            if col in categorical_cols:
                values, counts = np.unique(observed, return_counts=True)
                fill = values[np.argmax(counts)]
            else:
                fill = observed.mean()
            x_imputed[remaining, col] = fill

    def __call__(
        self,
        x,
        rows_to_impute: None | int | Iterable[int] | Iterable[str] = None,
        cols_to_impute: None | int | Iterable[int] | Iterable[str] = None,
        n_nearest_features: None | float | int = None,
        normalize: bool = True,
    ):
        """Imputes missing values in the input data.

        The method handles NumPy arrays and eager pandas or Polars DataFrames.

        Args:
            x: The input data matrix with missing values (NaNs).
                Can be a NumPy array or an eager pandas or Polars DataFrame.
            rows_to_impute: The rows to impute. The interpretation of this
                argument depends on the type of `x`.
                - If `x` is a NumPy array, this must be a list of integer indices.
                - If `x` is a pandas DataFrame, this must be a list of index labels.
                - If `x` is a Polars DataFrame, this must be a list of integer row positions.
                If None, all rows are considered for imputation. Defaults to None.
            cols_to_impute: The columns to impute. The interpretation of this
                argument depends on the type of `x`.
                - If `x` is a NumPy array, this must be a list of integer indices.
                - If `x` is a pandas or Polars DataFrame, this must be a list of column names.
                If None, all columns are considered for imputation. Defaults to None.
            n_nearest_features: The number of features to use for
                imputation. If it's an int, it's the absolute number of
                features. If it's a float, it's the fraction of features to
                use. If None, all features are used. Defaults to None.
            normalize: Whether to normalize numeric columns before imputation,
                then transform imputed values back to the original scale.
                Defaults to True.

        Returns:
            The imputed data matrix. The return type will match the input type
            (NumPy array, pandas DataFrame, or Polars DataFrame).
        """
        if is_polars_lazyframe(x):
            raise DataFillerTypeError("Polars LazyFrame input is not supported; call collect() before imputation.")

        is_pandas_df = isinstance(x, pd.DataFrame)
        is_polars_df = is_polars_dataframe(x)
        is_df = is_pandas_df or is_polars_df
        categorical_targets: dict[int, list] = {}
        encoded_feature_names: list[str] | None = None
        encoded_index_to_original: dict[int, str] = {}
        original_index = None
        original_columns = None
        main_column_indices = None
        original_dtypes = None
        normalize_cols = None
        norm_means = None
        norm_scales = None
        polars_metadata = None

        if is_pandas_df:
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
        elif is_polars_df:
            original_columns = list(x.columns)
            rows_to_impute = validate_polars_rows(rows_to_impute, x.height)
            cols_to_impute_df = polars_cols_to_indices(cols_to_impute, original_columns)
            cols_to_impute_processed = _process_to_impute(size=len(original_columns), to_impute=cols_to_impute_df)

            polars_metadata = encode_polars_dataframe(x)
            x = polars_metadata["data"]
            main_column_indices = polars_metadata["main_column_indices"]
            categorical_targets = polars_metadata["categorical_targets"]
            encoded_feature_names = polars_metadata["encoded_feature_names"]
            encoded_index_to_original = polars_metadata["encoded_index_to_original"]
            original_dtypes = polars_metadata["original_dtypes"]
            cols_to_impute = np.array([main_column_indices[idx] for idx in cols_to_impute_processed], dtype=np.int64)
        else:
            x = np.asarray(x)

        n_nearest_features = _validate_input(x, rows_to_impute, cols_to_impute, n_nearest_features)

        m, n = x.shape
        rows_to_impute = _process_to_impute(size=m, to_impute=rows_to_impute)
        cols_to_impute = _process_to_impute(size=n, to_impute=cols_to_impute)
        mask_rows_to_impute = _mask_index_to_impute(size=m, to_impute=rows_to_impute)
        categorical_cols = set(categorical_targets.keys())

        mask_nan = np.isnan(x)

        if normalize:
            if is_df:
                if is_polars_df:
                    assert polars_metadata is not None
                    normalize_cols = polars_metadata["numeric_main_indices"]
                else:
                    assert original_columns is not None and original_dtypes is not None
                    assert main_column_indices is not None
                    numeric_cols = []
                    for i, col in enumerate(original_columns):
                        dtype = original_dtypes[col]
                        if is_integer_dtype(dtype) or is_float_dtype(dtype):
                            numeric_cols.append(main_column_indices[i])
                    normalize_cols = np.array(numeric_cols, dtype=np.int64)
            else:
                normalize_cols = np.arange(n, dtype=np.int64)

            if normalize_cols.size:
                norm_means, norm_scales = self._masked_column_stats(x, mask_nan, normalize_cols)
                if not is_df:
                    x = x.copy()
                if normalize_cols.size == n and np.issubdtype(x.dtype, np.floating):
                    x -= norm_means
                    x /= norm_scales
                else:
                    x[:, normalize_cols] = (x[:, normalize_cols] - norm_means) / norm_scales

        if n_nearest_features is not None:
            if isinstance(self.scoring, str):
                scores = scoring(x, cols_to_impute, mask_nan)
            else:
                scores = self.scoring(x, cols_to_impute)
            self.imputation_features_ = {}
        else:
            scores = None
            self.imputation_features_ = None

        x_imputed = x.copy()

        if self.device is not None and type(self.regressor) is not FastRidge:
            warnings.warn(
                f"device={self.device!r} is ignored: the GPU path only supports the default FastRidge "
                f"regressor, so {type(self.regressor).__name__} runs on the CPU implementation.",
                UserWarning,
                stacklevel=2,
            )
            self._gpu_backend = None
        else:
            self._gpu_backend = GramBackend(self.device) if self.device is not None else None
        try:
            for i, col in enumerate(tqdm(cols_to_impute, leave=False, disable=(not self.verbose))):
                self._impute_col(
                    x,
                    x_imputed,
                    col,
                    mask_nan,
                    mask_rows_to_impute,
                    n_nearest_features,
                    scores,
                    i,
                    categorical_cols,
                )
        finally:
            if self._gpu_backend is not None:
                self._gpu_backend.release()
                self._gpu_backend = None

        if normalize and normalize_cols is not None and normalize_cols.size:
            if normalize_cols.size == n and np.issubdtype(x_imputed.dtype, np.floating):
                x_imputed *= norm_scales
                x_imputed += norm_means
            else:
                x_imputed[:, normalize_cols] = x_imputed[:, normalize_cols] * norm_scales + norm_means

        if self.fallback == "simple":
            self._apply_fallback(x_imputed, mask_nan, mask_rows_to_impute, cols_to_impute, categorical_cols)

        if is_df and self.imputation_features_ is not None:
            assert encoded_feature_names is not None
            self.imputation_features_ = {
                encoded_index_to_original.get(col, encoded_feature_names[col]): [
                    encoded_index_to_original.get(feature, encoded_feature_names[feature]) for feature in features
                ]
                for col, features in self.imputation_features_.items()
            }

        if is_pandas_df:
            assert isinstance(original_index, pd.Index) and isinstance(original_columns, pd.Index)
            assert main_column_indices is not None and original_dtypes is not None
            return self._decode_dataframe(
                x_imputed=x_imputed,
                original_index=original_index,
                original_columns=original_columns,
                main_column_indices=main_column_indices,
                categorical_targets=categorical_targets,
                original_dtypes=original_dtypes,
            )
        if is_polars_df:
            assert polars_metadata is not None
            return decode_polars_dataframe(x_imputed, polars_metadata)

        return x_imputed
