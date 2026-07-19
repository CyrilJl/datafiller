from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin

from ..exceptions import DataFillerTypeError, DataFillerValueError
from ..multivariate._polars import is_polars_dataframe, is_polars_lazyframe
from ..multivariate.imputer import MultivariateImputer
from ._utils import interpolate_small_gaps


class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values in time series data.

    This class wraps the :class:`MultivariateImputer` to specifically handle
    time series data in pandas DataFrames. It automatically creates lagged and
    lead features based on the time series index, then uses these new
    features to impute missing values.

    Args:
        lags (Iterable[int], optional): An iterable of integers specifying
            the lags and leads to create as autoregressive features. Positive
            integers create lags (e.g., `t-1`), and negative integers create
            leads (e.g., `t+1`). Defaults to `(1,)`.
        regressor (RegressorMixin, optional): A scikit-learn compatible
            regressor used for numeric targets. Defaults to ``FastRidge``.
        classifier (ClassifierMixin, optional): A scikit-learn compatible
            classifier used for categorical or string targets. Defaults to
            ``DecisionTreeClassifier(max_depth=4)``.
        min_samples_train (int, optional): The minimum number of samples
            required to train a model. Defaults to `None`, which resolves to
            20 (see :class:`~datafiller.multivariate.imputer.MultivariateImputer`).
        fallback (str or None, optional): What to do with cells no model could
            impute. ``"simple"`` (default) fills them with the column mean (mode
            for categoricals); ``None`` leaves them as NaN.
        rng (int, optional): A seed for the random number generator. This is
            used for reproducible feature sampling when `n_nearest_features`
            is not None. Defaults to None.
        verbose (int, optional): The verbosity level. Defaults to 0.
        scoring (str or callable, optional): The scoring function to use for
            feature selection. If 'default', the default scoring function is
            used. If a callable, it must take two arguments (the data matrix
            and the columns to impute) and return a score matrix.
            Defaults to 'default'.
        interpolate_gaps_less_than (int, optional): The maximum length of
            gaps to interpolate linearly. If None, no linear interpolation is
            performed. Defaults to None.
        device (str, optional): Device used to solve the default ridge
            models, e.g. ``"cuda"``. Requires the optional PyTorch dependency
            (``pip install datafiller[gpu]``). If None (default), the pure
            NumPy/Numba CPU path is used. See
            :class:`~datafiller.multivariate.imputer.MultivariateImputer`.
        add_time_features (bool, optional): Whether to add deterministic time
            features before model-based imputation. These features are fully
            observed after reindexing, which helps fill contiguous missing
            timestamp blocks. Defaults to True.
        time_column (str, optional): Name of the Date or Datetime column that
            represents time for a Polars DataFrame. Required for Polars input;
            pandas input continues to use its DatetimeIndex. Defaults to None.

    Attributes:
        imputation_features_ (dict or None): A dictionary mapping each imputed
            column to the features used to impute it. This attribute is only
            populated when `n_nearest_features` is not None. The keys and
            values are the column names, which will include the lagged/lead
            features created during the imputation process.

    .. code-block:: python

        import pandas as pd
        import numpy as np
        from datafiller import TimeSeriesImputer

        # Create a time series DataFrame with missing values
        rng = pd.date_range('2020-01-01', periods=10, freq='D')
        data = {'value': [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10]}
        df = pd.DataFrame(data, index=rng)

        # Create a time series imputer and fill missing values
        ts_imputer = TimeSeriesImputer(lags=[1, -1])
        df_imputed = ts_imputer(df)

        print(df_imputed)
    """

    def __init__(
        self,
        lags: Iterable[int] = (1,),
        regressor: RegressorMixin | None = None,
        classifier: ClassifierMixin | None = None,
        min_samples_train: int | None = None,
        fallback: str | None = "simple",
        rng: int | None = None,
        verbose: int = 0,
        scoring: str | Callable = "default",
        interpolate_gaps_less_than: int | None = None,
        add_time_features: bool = True,
        device: str | None = None,
        time_column: str | None = None,
    ):
        if not isinstance(lags, Iterable) or not all(isinstance(i, int) for i in lags):
            raise DataFillerValueError("lags must be an iterable of integers.")
        if 0 in lags:
            raise DataFillerValueError("lags cannot contain 0.")
        if time_column is not None and not isinstance(time_column, str):
            raise DataFillerValueError("time_column must be a string or None.")
        self.lags = lags
        self.regressor = regressor
        self.classifier = classifier
        self.min_samples_train = min_samples_train
        self.fallback = fallback
        self.rng = rng
        self.verbose = verbose
        self.scoring = scoring
        self.interpolate_gaps_less_than = interpolate_gaps_less_than
        self.add_time_features = add_time_features
        self.device = device
        self.time_column = time_column
        self._build_multivariate_imputer()
        self.imputation_features_ = None

    def _build_multivariate_imputer(self) -> None:
        self.multivariate_imputer = MultivariateImputer(
            regressor=self.regressor,
            classifier=self.classifier,
            verbose=self.verbose,
            min_samples_train=self.min_samples_train,
            fallback=self.fallback,
            rng=self.rng,
            scoring=self.scoring,
            device=self.device,
        )

    def fit(self, X, y: None = None) -> "TimeSeriesImputer":
        """No-op fit for sklearn compatibility."""
        return self

    def transform(self, X):
        """Impute missing values in X using stored configuration."""
        return self(X)

    def set_params(self, **params) -> "TimeSeriesImputer":
        """Set parameters and refresh dependent objects."""
        if "time_column" in params and params["time_column"] is not None and not isinstance(params["time_column"], str):
            raise DataFillerValueError("time_column must be a string or None.")
        rebuild_keys = {
            "regressor",
            "classifier",
            "min_samples_train",
            "fallback",
            "rng",
            "verbose",
            "scoring",
            "device",
        }
        rebuild = any(key in params for key in rebuild_keys)

        super().set_params(**params)

        if rebuild:
            self._build_multivariate_imputer()

        return self

    @staticmethod
    def _infer_frequency(index: pd.DatetimeIndex) -> object:
        """Infer the base frequency, allowing regular gaps in the index."""
        if index.freq is not None:
            return index.freq

        if len(index) < 2:
            raise DataFillerValueError("DataFrame index must have a frequency or at least two timestamps to infer one.")
        if len(index) >= 3:
            inferred = pd.infer_freq(index)
            if inferred is not None:
                return inferred
        if not index.is_monotonic_increasing:
            raise DataFillerValueError("DataFrame index must be sorted in increasing order.")
        if index.has_duplicates:
            raise DataFillerValueError("DataFrame index must not contain duplicate timestamps.")

        timestamps_ns = index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        deltas = np.diff(timestamps_ns)
        positive_deltas = deltas[deltas > 0]
        if not positive_deltas.size:
            raise DataFillerValueError("DataFrame index frequency could not be inferred.")

        base_delta = positive_deltas.min()
        if np.any(positive_deltas % base_delta != 0):
            raise DataFillerValueError("DataFrame index frequency could not be inferred from irregular timestamp gaps.")
        return pd.Timedelta(base_delta, unit="ns")

    @classmethod
    def _regularize_index(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex a time series to its complete regular timestamp grid."""
        assert isinstance(df.index, pd.DatetimeIndex)
        freq = cls._infer_frequency(df.index)
        full_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq, name=df.index.name)
        if len(full_index) == len(df.index) and full_index.equals(df.index):
            return df
        return df.reindex(full_index)

    @staticmethod
    def _make_time_features(index: pd.DatetimeIndex, reserved_names: Iterable[str] = ()) -> pd.DataFrame:
        """Create low-cost, fully observed calendar features."""
        elapsed = ((index - index[0]) / pd.Timedelta(days=1)).to_numpy(dtype=np.float32)
        if elapsed.size and elapsed[-1] != 0:
            trend = elapsed / elapsed[-1]
        else:
            trend = np.zeros(len(index), dtype=np.float32)

        hour = index.hour.to_numpy(dtype=np.float32) + index.minute.to_numpy(dtype=np.float32) / 60.0  # ty: ignore[unresolved-attribute]
        day_angle = np.float32(2.0 * np.pi) * hour / np.float32(24.0)
        week_angle = np.float32(2.0 * np.pi) * index.dayofweek.to_numpy(dtype=np.float32) / np.float32(7.0)  # ty: ignore[unresolved-attribute]

        base_features = {
            "__time_trend": trend.astype(np.float32, copy=False),
            "__time_day_sin": np.sin(day_angle).astype(np.float32, copy=False),
            "__time_day_cos": np.cos(day_angle).astype(np.float32, copy=False),
            "__time_week_sin": np.sin(week_angle).astype(np.float32, copy=False),
            "__time_week_cos": np.cos(week_angle).astype(np.float32, copy=False),
        }
        used_names = set(reserved_names)
        features = {}
        for name, values in base_features.items():
            feature_name = name
            suffix = 1
            while feature_name in used_names:
                feature_name = f"{name}_{suffix}"
                suffix += 1
            used_names.add(feature_name)
            features[feature_name] = values

        return pd.DataFrame(features, index=index)

    def _polars_to_pandas(self, df) -> tuple[pd.DataFrame, dict]:
        """Materialize a numeric Polars time series on a pandas DatetimeIndex."""
        import polars as pl

        if self.time_column is None:
            raise DataFillerValueError("time_column must be set when TimeSeriesImputer receives a Polars DataFrame.")
        if self.time_column not in df.columns:
            raise DataFillerValueError(f"time_column {self.time_column!r} was not found in the Polars DataFrame.")

        time_series = df.get_column(self.time_column)
        if time_series.dtype.base_type() not in {pl.Date, pl.Datetime}:
            raise DataFillerTypeError(
                f"Polars time_column must have Date or Datetime dtype, got {time_series.dtype} for "
                f"{self.time_column!r}."
            )
        if time_series.null_count():
            raise DataFillerValueError("Polars time_column cannot contain null values.")

        data_columns = [column for column in df.columns if column != self.time_column]
        if not data_columns:
            raise DataFillerValueError(
                "A Polars time series must contain at least one data column besides time_column."
            )
        unsupported = [column for column in data_columns if not df.schema[column].is_numeric()]
        if unsupported:
            raise DataFillerValueError(
                f"TimeSeriesImputer requires numeric data columns; unsupported columns: {unsupported}"
            )

        index = pd.DatetimeIndex(time_series.to_list(), name=self.time_column)
        data = {column: df.get_column(column).to_numpy() for column in data_columns}
        pandas_df = pd.DataFrame(data, index=index)
        metadata = {
            "columns": list(df.columns),
            "data_columns": data_columns,
            "dtypes": dict(df.schema),
            "original_timestamps": set(index),
            "null_timestamps": {
                column: set(index[df.get_column(column).is_null().to_numpy()]) for column in data_columns
            },
        }
        return pandas_df, metadata

    def _pandas_to_polars(self, df: pd.DataFrame, metadata: dict):
        """Restore an imputed pandas working frame to its original Polars schema."""
        import polars as pl

        timestamps = list(df.index)
        output = {}
        for column in metadata["columns"]:
            dtype = metadata["dtypes"][column]
            if column == self.time_column:
                time_values = (
                    [timestamp.date() for timestamp in timestamps] if dtype.base_type() == pl.Date else timestamps
                )
                output[column] = pl.Series(column, time_values, dtype=dtype)
                continue

            values = df[column].to_numpy()
            if dtype.is_integer():
                decoded = [None if np.isnan(value) else int(np.rint(value)) for value in values]
            else:
                null_timestamps = metadata["null_timestamps"][column]
                original_timestamps = metadata["original_timestamps"]
                decoded = [
                    None
                    if np.isnan(value) and (timestamp in null_timestamps or timestamp not in original_timestamps)
                    else value
                    for timestamp, value in zip(timestamps, values, strict=True)
                ]
            output[column] = pl.Series(column, decoded, dtype=dtype)
        return pl.DataFrame(output).select(metadata["columns"])

    @staticmethod
    def _polars_rows_to_indices(rows_to_impute, index: pd.DatetimeIndex):
        """Resolve Polars row positions or timestamp values on the regularized grid."""
        if rows_to_impute is None or isinstance(rows_to_impute, (int, np.integer)):
            return rows_to_impute

        if isinstance(rows_to_impute, str) or not isinstance(rows_to_impute, Iterable):
            values = [rows_to_impute]
        else:
            values = list(rows_to_impute)
        if all(isinstance(value, (int, np.integer)) for value in values):
            return values

        timestamps = pd.DatetimeIndex(pd.to_datetime(values))
        positions = index.get_indexer(timestamps)
        if np.any(positions == -1):
            missing = [value for value, position in zip(values, positions, strict=True) if position == -1]
            raise DataFillerValueError(f"Timestamps not found in the regularized time grid: {missing}")
        return positions

    def __call__(
        self,
        df,
        rows_to_impute: None | int | Iterable[int] = None,
        cols_to_impute: None | int | str | Iterable[int | str] = None,
        n_nearest_features: None | float | int = None,
        before: object = None,
        after: object = None,
    ):
        """Imputes missing values in a time series DataFrame.

        Args:
            df: A pandas DataFrame with a DatetimeIndex, or an eager Polars
                DataFrame with the Date/Datetime column configured by
                ``time_column``. If the time axis has no explicit frequency, a
                regular one is inferred from the timestamps and any missing
                timestamps inside the observed range are reinserted as rows
                to impute.
            rows_to_impute: The rows to impute. Can be an iterable of
                integer positions, timestamp values, a pandas DatetimeIndex,
                or None. Polars integer positions refer to the regularized
                output grid. If None, all rows are considered. Defaults to None.
            cols_to_impute: The indices or names of columns
                to impute. If None, all columns are considered. Defaults to None.
            n_nearest_features: The number of features to use for
                imputation. If it's an int, it's the absolute number of
                features. If it's a float, it's the fraction of features to
                use. If None, all features are used. Defaults to None.
            before: A timestamp-like object. If specified, only rows
                before this timestamp are imputed. Can be anything that can be
                parsed by ``lambda x: pd.to_datetime(str(x))``. Defaults to None.
            after: A timestamp-like object. If specified, only rows
                after this timestamp are imputed. Can be anything that can be
                parsed by ``lambda x: pd.to_datetime(str(x))``. Defaults to None.

        Returns:
            The imputed DataFrame, matching the input dataframe implementation.

        Raises:
            TypeError: If the input is unsupported or its time axis is invalid.
            ValueError: If no regular frequency can be inferred from the
                index (e.g. unsorted or duplicated timestamps, or irregular
                gaps), or if the columns are not numeric.
        """
        if is_polars_lazyframe(df):
            raise DataFillerTypeError("Polars LazyFrame input is not supported; call collect() before imputation.")
        is_polars_df = is_polars_dataframe(df)
        polars_metadata = None
        if is_polars_df:
            polars_rows_to_impute = rows_to_impute
            df, polars_metadata = self._polars_to_pandas(df)
        else:
            if not isinstance(df, pd.DataFrame):
                raise DataFillerTypeError("Input must be a pandas or eager Polars DataFrame.")
            if not isinstance(df.index, pd.DatetimeIndex):
                raise DataFillerTypeError("DataFrame index must be a DatetimeIndex.")

        df = self._regularize_index(df)
        assert isinstance(df.index, pd.DatetimeIndex)

        if is_polars_df:
            rows_to_impute = self._polars_rows_to_indices(polars_rows_to_impute, df.index)
            if cols_to_impute is not None:
                requested_columns = (
                    [cols_to_impute]
                    if isinstance(cols_to_impute, str) or not isinstance(cols_to_impute, Iterable)
                    else list(cols_to_impute)
                )
                if not all(isinstance(column, str) for column in requested_columns):
                    raise DataFillerValueError("cols_to_impute must contain column names for a Polars DataFrame.")
                if self.time_column in requested_columns:
                    raise DataFillerValueError("The Polars time_column cannot be imputed.")

        if self.interpolate_gaps_less_than is not None:
            df = df.copy()
            for col in df.columns:
                df[col] = interpolate_small_gaps(df[col], self.interpolate_gaps_less_than)

        original_cols = df.columns
        n_original_cols = len(original_cols)

        values = df.to_numpy()
        if not np.issubdtype(values.dtype, np.floating):
            try:
                values = values.astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise DataFillerValueError("TimeSeriesImputer requires numeric columns.") from exc

        # Create autoregressive (and optional calendar) features directly in a
        # preallocated matrix instead of concatenating shifted DataFrames.
        lags = list(self.lags)
        feature_names = list(original_cols)
        for lag in lags:
            feature_names.extend(f"{col}_lag_{lag}" for col in original_cols)
        if self.add_time_features:
            assert isinstance(df.index, pd.DatetimeIndex)
            time_features = self._make_time_features(df.index, reserved_names=feature_names)
            feature_names.extend(time_features.columns)
        else:
            time_features = None

        matrix = np.empty((len(df), len(feature_names)), dtype=values.dtype)
        matrix[:, :n_original_cols] = values
        for i, lag in enumerate(lags):
            start = n_original_cols * (i + 1)
            block = matrix[:, start : start + n_original_cols]
            n_kept_rows = max(0, len(df) - abs(lag))
            if lag > 0:
                block[: len(df) - n_kept_rows] = np.nan
                block[len(df) - n_kept_rows :] = values[:n_kept_rows]
            else:
                block[:n_kept_rows] = values[len(df) - n_kept_rows :]
                block[n_kept_rows:] = np.nan
        if time_features is not None:
            matrix[:, len(feature_names) - time_features.shape[1] :] = time_features.to_numpy(dtype=values.dtype)

        # Equivalent of dropna(how="all", axis=1) on the feature frame.
        all_nan_cols = np.isnan(matrix).all(axis=0)
        if all_nan_cols.any():
            keep = ~all_nan_cols
            matrix = np.ascontiguousarray(matrix[:, keep])
            feature_names = [name for name, keep_col in zip(feature_names, keep, strict=True) if keep_col]
        feature_index = pd.Index(feature_names)

        # Process cols_to_impute
        if cols_to_impute is None:
            cols_to_impute_indices = np.arange(n_original_cols)
        else:
            if isinstance(cols_to_impute, (int, str)):
                cols_to_impute = [cols_to_impute]

            indices = []
            for c in cols_to_impute:
                if isinstance(c, int):
                    indices.append(c)
                elif isinstance(c, str):
                    indices.append(original_cols.get_loc(c))
                else:
                    raise DataFillerValueError("cols_to_impute must be an int, str, or an iterable of ints or strs.")
            cols_to_impute_indices = np.array(indices)

        # Process rows_to_impute
        if rows_to_impute is not None:
            if isinstance(rows_to_impute, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)):
                rows_to_impute = df.index.get_indexer(rows_to_impute)
            elif isinstance(rows_to_impute, int):
                rows_to_impute = [rows_to_impute]
        elif rows_to_impute is None:
            if before is not None or after is not None:
                mask = pd.Series(True, index=df.index)
                if before is not None and (before_timestamp := pd.to_datetime(str(before))):
                    mask &= df.index < before_timestamp
                if after is not None and (after_timestamp := pd.to_datetime(str(after))):
                    mask &= df.index > after_timestamp
                rows_to_impute = np.where(mask)[0]

        # Impute the data
        imputed_data = self.multivariate_imputer(
            matrix,
            rows_to_impute=rows_to_impute,
            cols_to_impute=cols_to_impute_indices,
            n_nearest_features=n_nearest_features,
        )
        self.imputation_features_ = self.multivariate_imputer.imputation_features_

        if self.imputation_features_ is not None:
            self.imputation_features_ = {
                feature_index[col]: feature_index[features].tolist()
                for col, features in self.imputation_features_.items()
            }

        # Return a DataFrame with the same columns as the original. Slice the
        # array before wrapping so pandas does not copy the full feature matrix.
        if feature_index.is_unique:
            positions = feature_index.get_indexer(original_cols)
            if (positions >= 0).all():
                result = pd.DataFrame(imputed_data[:, positions], index=df.index, columns=original_cols)
                if is_polars_df:
                    assert polars_metadata is not None
                    return self._pandas_to_polars(result, polars_metadata)
                return result
        imputed_df = pd.DataFrame(imputed_data, index=df.index, columns=feature_index)
        result = imputed_df[original_cols]
        if is_polars_df:
            assert polars_metadata is not None
            return self._pandas_to_polars(result, polars_metadata)
        return result
