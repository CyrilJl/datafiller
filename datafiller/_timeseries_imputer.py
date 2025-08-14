import pandas as pd
import numpy as np
from typing import Iterable, Union
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression

from ._multivariate_imputer import MultivariateImputer

class TimeSeriesImputer:
    """
    A time series imputer that uses autoregressive features to fill missing values.

    This class wraps the `MultivariateImputer` and is designed to work with
    pandas DataFrames that have a `DatetimeIndex` with a defined frequency.
    It creates lagged and lead features and uses them to impute missing values
    in the original columns.

    Parameters
    ----------
    lags : Iterable[int], default=(1,)
        An iterable of integers specifying the lags and leads to create as
        autoregressive features. Positive integers create lags (e.g., `t-1`),
        and negative integers create leads (e.g., `t+1`).
    estimator : RegressorMixin, default=LinearRegression()
        A scikit-learn compatible estimator to use for imputation.
    verbose : int, default=0
        The verbosity level.
    min_samples_train : int, default=50
        The minimum number of samples required to train a model.
    """

    def __init__(
        self,
        lags: Iterable[int] = (1,),
        estimator: RegressorMixin = LinearRegression(),
        verbose: int = 0,
        min_samples_train: int = 50,
    ):
        if not isinstance(lags, Iterable) or not all(isinstance(i, int) for i in lags):
            raise ValueError("lags must be an iterable of integers.")
        if 0 in lags:
            raise ValueError("lags cannot contain 0.")
        self.lags = lags
        self.multivariate_imputer = MultivariateImputer(
            estimator=estimator, verbose=verbose, min_samples_train=min_samples_train
        )

    def __call__(
        self,
        df: pd.DataFrame,
        rows_to_impute: Union[None, int, Iterable[int]] = None,
        cols_to_impute: Union[None, int, str, Iterable[Union[int, str]]] = None,
        n_nearest_features: Union[None, float, int] = None,
    ):
        """
        Imputes missing values in a time series DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame with a `DatetimeIndex` and missing values (NaNs).
        rows_to_impute : None | int | Iterable[int], default=None
            The indices of rows to impute. If None, all rows are considered.
        cols_to_impute : None | int | str | Iterable[Union[int, str]], default=None
            The indices or names of columns to impute. If None, all columns
            are considered.
        n_nearest_features : None | float | int, default=None
            The number of features to use for imputation. If it's an int, it's
            the absolute number of features. If it's a float, it's the
            fraction of features to use. If None, all features are used.

        Returns
        -------
        pd.DataFrame
            The imputed DataFrame with the same columns as the original.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        if df.index.freq is None:
            raise ValueError("DataFrame index must have a frequency.")

        original_cols = df.columns
        n_original_cols = len(original_cols)

        # Create autoregressive features
        df_with_lags = df.copy()
        for lag in self.lags:
            shifted = df.shift(lag)
            shifted.columns = [f"{col}_lag_{lag}" for col in original_cols]
            df_with_lags = pd.concat([df_with_lags, shifted], axis=1)

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

        # Impute the data
        imputed_data = self.multivariate_imputer(
            df_with_lags.values,
            rows_to_impute=rows_to_impute,
            cols_to_impute=cols_to_impute_indices,
            n_nearest_features=n_nearest_features
        )

        # Return a DataFrame with the same columns as the original
        imputed_df = pd.DataFrame(imputed_data, index=df.index, columns=df_with_lags.columns)
        return imputed_df[original_cols]
