from typing import Iterable, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin

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
            required to train a model. Defaults to `None`, which means that a
            model will be trained if at least one sample is available.
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
        add_time_features (bool, optional): Whether to add deterministic time
            features before model-based imputation. These features are fully
            observed after reindexing, which helps fill contiguous missing
            timestamp blocks. Defaults to True.

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
        rng: Union[int, None] = None,
        verbose: int = 0,
        scoring: Union[str, callable] = "default",
        interpolate_gaps_less_than: int = None,
        add_time_features: bool = True,
    ):
        if not isinstance(lags, Iterable) or not all(isinstance(i, int) for i in lags):
            raise ValueError("lags must be an iterable of integers.")
        if 0 in lags:
            raise ValueError("lags cannot contain 0.")
        self.lags = lags
        self.regressor = regressor
        self.classifier = classifier
        self.min_samples_train = min_samples_train
        self.rng = rng
        self.verbose = verbose
        self.scoring = scoring
        self.interpolate_gaps_less_than = interpolate_gaps_less_than
        self.add_time_features = add_time_features
        self._build_multivariate_imputer()
        self.imputation_features_ = None

    def _build_multivariate_imputer(self) -> None:
        min_samples_train = 1 if self.min_samples_train is None else self.min_samples_train
        self.multivariate_imputer = MultivariateImputer(
            regressor=self.regressor,
            classifier=self.classifier,
            verbose=self.verbose,
            min_samples_train=min_samples_train,
            rng=self.rng,
            scoring=self.scoring,
        )

    def fit(self, X: pd.DataFrame, y: None = None) -> "TimeSeriesImputer":
        """No-op fit for sklearn compatibility."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in X using stored configuration."""
        return self(X)

    def set_params(self, **params) -> "TimeSeriesImputer":
        """Set parameters and refresh dependent objects."""
        rebuild_keys = {"regressor", "classifier", "min_samples_train", "rng", "verbose", "scoring"}
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
            raise ValueError("DataFrame index must have a frequency or at least two timestamps to infer one.")
        if len(index) >= 3:
            inferred = pd.infer_freq(index)
            if inferred is not None:
                return inferred
        if not index.is_monotonic_increasing:
            raise ValueError("DataFrame index must be sorted in increasing order.")
        if index.has_duplicates:
            raise ValueError("DataFrame index must not contain duplicate timestamps.")

        timestamps_ns = index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        deltas = np.diff(timestamps_ns)
        positive_deltas = deltas[deltas > 0]
        if not positive_deltas.size:
            raise ValueError("DataFrame index frequency could not be inferred.")

        base_delta = positive_deltas.min()
        if np.any(positive_deltas % base_delta != 0):
            raise ValueError("DataFrame index frequency could not be inferred from irregular timestamp gaps.")
        return pd.Timedelta(base_delta, unit="ns")

    @classmethod
    def _regularize_index(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex a time series to its complete regular timestamp grid."""
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

        hour = index.hour.to_numpy(dtype=np.float32) + index.minute.to_numpy(dtype=np.float32) / 60.0
        day_angle = np.float32(2.0 * np.pi) * hour / np.float32(24.0)
        week_angle = np.float32(2.0 * np.pi) * index.dayofweek.to_numpy(dtype=np.float32) / np.float32(7.0)

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

    def __call__(
        self,
        df: pd.DataFrame,
        rows_to_impute: Union[None, int, Iterable[int]] = None,
        cols_to_impute: Union[None, int, str, Iterable[Union[int, str]]] = None,
        n_nearest_features: Union[None, float, int] = None,
        before: object = None,
        after: object = None,
    ) -> pd.DataFrame:
        """Imputes missing values in a time series DataFrame.

        Args:
            df: The input DataFrame with a `DatetimeIndex` and missing
                values (NaNs). The index must have a defined frequency.
            rows_to_impute: The rows to impute. Can be an iterable of
                integer indices, a pandas DatetimeIndex, or None. If None,
                all rows are considered. Defaults to None.
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
            The imputed DataFrame with the same columns as the original.

        Raises:
            TypeError: If the input is not a pandas DataFrame or if the index
                is not a DatetimeIndex.
            ValueError: If the DataFrame's index does not have a frequency.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")

        df = self._regularize_index(df)

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
                raise ValueError("TimeSeriesImputer requires numeric columns.") from exc

        # Create autoregressive (and optional calendar) features directly in a
        # preallocated matrix instead of concatenating shifted DataFrames.
        lags = list(self.lags)
        feature_names = list(original_cols)
        for lag in lags:
            feature_names.extend(f"{col}_lag_{lag}" for col in original_cols)
        if self.add_time_features:
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
                    raise ValueError("cols_to_impute must be an int, str, or an iterable of ints or strs.")
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
                return pd.DataFrame(imputed_data[:, positions], index=df.index, columns=original_cols)
        imputed_df = pd.DataFrame(imputed_data, index=df.index, columns=feature_index)
        return imputed_df[original_cols]
