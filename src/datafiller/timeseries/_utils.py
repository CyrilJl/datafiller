import numpy as np
import pandas as pd
from numba import njit, prange

from ..exceptions import DataFillerTypeError


@njit(boundscheck=False, cache=True, parallel=True)
def build_lag_matrix(values: np.ndarray, lags: np.ndarray, time_features: np.ndarray) -> np.ndarray:
    """Assemble the autoregressive feature matrix in a single pass.

    Column layout is the original series, then one block per entry of `lags`
    (positive shifts back in time, negative forward), then the calendar
    features. Writing row by row keeps every store contiguous, where filling
    one column block at a time walks the output with a large stride.

    Args:
        values: The observed series, shape `(n_rows, n_series)`.
        lags: Shifts to materialize. Out-of-range positions become NaN.
        time_features: Fully observed calendar features, shape `(n_rows, n_time)`.

    Returns:
        The feature matrix, shape `(n_rows, n_series * (1 + len(lags)) + n_time)`.
    """
    n_rows, n_series = values.shape
    n_lags = len(lags)
    n_time = time_features.shape[1]
    out = np.empty((n_rows, n_series * (1 + n_lags) + n_time), dtype=values.dtype)
    for i in prange(n_rows):  # ty: ignore[not-iterable]
        target = out[i]
        source = values[i]
        for j in range(n_series):
            target[j] = source[j]
        for t in range(n_lags):
            offset = n_series * (t + 1)
            shifted = i - lags[t]
            if 0 <= shifted < n_rows:
                lagged = values[shifted]
                for j in range(n_series):
                    target[offset + j] = lagged[j]
            else:
                for j in range(n_series):
                    target[offset + j] = np.nan
        offset = n_series * (1 + n_lags)
        calendar = time_features[i]
        for j in range(n_time):
            target[offset + j] = calendar[j]
    return out


def interpolate_small_gaps(series: pd.Series, n: int) -> pd.Series:
    """Interpolate missing values (NaN) in a Pandas Series,
    but only for gaps of length n or less.

    Parameters:
        series (pd.Series): The Series containing missing values.
        n (int): The maximum length of gaps to interpolate.

    Returns:
        pd.Series: The Series with small gaps interpolated.
    """
    if not isinstance(n, int):
        raise DataFillerTypeError("n must be an int")
    is_nan = series.isna()
    gaps = (is_nan != is_nan.shift()).cumsum()
    mask = series.groupby(gaps).transform("size") <= n
    return series.interpolate().where(mask, series)
