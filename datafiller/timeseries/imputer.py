from typing import Iterable, Union

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

from ..estimators.ridge import FastRidge
from ..multivariate.imputer import MultivariateImputer
from ._utils import interpolate_small_gaps


class TimeSeriesImputer:
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
        estimator (RegressorMixin, optional): A scikit-learn compatible
            estimator to use for imputation. Defaults to `LinearRegression()`.
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
        estimator: RegressorMixin = FastRidge(),
        min_samples_train: int | None = None,
        rng: Union[int, None] = None,
        verbose: int = 0,
        scoring: Union[str, callable] = "default",
        interpolate_gaps_less_than: int = None,
    ):
        if not isinstance(lags, Iterable) or not all(isinstance(i, int) for i in lags):
            raise ValueError("lags must be an iterable of integers.")
        if 0 in lags:
            raise ValueError("lags cannot contain 0.")
        self.lags = lags
        self.interpolate_gaps_less_than = interpolate_gaps_less_than
        if min_samples_train is None:
            min_samples_train = 1
        self.multivariate_imputer = MultivariateImputer(
            estimator=estimator,
            verbose=verbose,
            min_samples_train=min_samples_train,
            rng=rng,
            scoring=scoring,
        )
        self.imputation_features_ = None

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
        if df.index.freq is None:
            raise ValueError("DataFrame index must have a frequency.")

        if self.interpolate_gaps_less_than is not None:
            for col in df.columns:
                df[col] = interpolate_small_gaps(df[col], self.interpolate_gaps_less_than)

        original_cols = df.columns
        n_original_cols = len(original_cols)

        # Create autoregressive features
        df_with_lags = df.copy()
        for lag in self.lags:
            shifted = df.shift(lag)
            shifted.columns = [f"{col}_lag_{lag}" for col in original_cols]
            df_with_lags = pd.concat([df_with_lags, shifted], axis=1)
        df_with_lags = df_with_lags.dropna(how="all", axis=1)

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
                before_timestamp = pd.to_datetime(str(before)) if before is not None else None
                after_timestamp = pd.to_datetime(str(after)) if after is not None else None

                mask = pd.Series(True, index=df.index)
                if before_timestamp:
                    mask &= df.index < before_timestamp
                if after_timestamp:
                    mask &= df.index > after_timestamp
                rows_to_impute = np.where(mask)[0]

        # Impute the data
        imputed_data = self.multivariate_imputer(
            df_with_lags.values,
            rows_to_impute=rows_to_impute,
            cols_to_impute=cols_to_impute_indices,
            n_nearest_features=n_nearest_features,
        )
        self.imputation_features_ = self.multivariate_imputer.imputation_features_

        if self.imputation_features_ is not None:
            self.imputation_features_ = {
                df_with_lags.columns[col]: df_with_lags.columns[features].tolist()
                for col, features in self.imputation_features_.items()
            }

        # Return a DataFrame with the same columns as the original
        imputed_df = pd.DataFrame(imputed_data, index=df.index, columns=df_with_lags.columns)
        return imputed_df[original_cols]
