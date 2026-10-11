import numpy as np
import pandas as pd
from numba import njit, prange

from ..exceptions import DataFillerTypeError


@njit(boundscheck=False, cache=True, parallel=True)
def build_lag_matrix(
    values: np.ndarray, sources: np.ndarray, shifts: np.ndarray, time_features: np.ndarray
) -> np.ndarray:
    """Assemble the autoregressive feature matrix in a single pass.

    Column layout is the original series, then one column per `(sources[f],
    shifts[f])` pair, then the calendar features. Lag feature `f` reads
    `values[i - shifts[f], sources[f]]` (positive shifts back in time, negative
    forward). Writing row by row keeps every store contiguous, where filling
    one column at a time walks the output with a large stride.

    Args:
        values: The observed series, shape `(n_rows, n_series)`.
        sources: Column of `values` each lag feature reads.
        shifts: Shift of each lag feature, in steps. Out-of-range positions become NaN.
        time_features: Fully observed calendar features, shape `(n_rows, n_time)`.

    Returns:
        The feature matrix, shape `(n_rows, n_series + len(shifts) + n_time)`.
    """
    n_rows, n_series = values.shape
    n_lagged = len(shifts)
    n_time = time_features.shape[1]

    # Coalesce consecutive features that read adjacent columns at the same shift
    # into runs, so the uniform layout (every column per lag) copies whole row
    # blocks instead of gathering one value at a time.
    run_first = np.empty(n_lagged, dtype=np.int64)
    run_length = np.empty(n_lagged, dtype=np.int64)
    n_runs = 0
    f = 0
    while f < n_lagged:
        g = f + 1
        while g < n_lagged and shifts[g] == shifts[f] and sources[g] == sources[g - 1] + 1:
            g += 1
        run_first[n_runs] = f
        run_length[n_runs] = g - f
        n_runs += 1
        f = g

    out = np.empty((n_rows, n_series + n_lagged + n_time), dtype=values.dtype)
    for i in prange(n_rows):
        target = out[i]
        source = values[i]
        for j in range(n_series):
            target[j] = source[j]
        for r in range(n_runs):
            first = run_first[r]
            offset = n_series + first
            shifted = i - shifts[first]
            length = run_length[r]
            if 0 <= shifted < n_rows:
                column = sources[first]
                target[offset : offset + length] = values[shifted, column : column + length]
            else:
                target[offset : offset + length] = np.nan
        offset = n_series + n_lagged
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
