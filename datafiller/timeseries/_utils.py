import pandas as pd


def check_params(param, types):
    if not isinstance(param, types):
        raise TypeError(f"param must be of type {types}")


def interpolate_small_gaps(series: pd.Series, n: int) -> pd.Series:
    """Interpolate missing values (NaN) in a Pandas Series,
    but only for gaps of length n or less.

    Parameters:
        series (pd.Series): The Series containing missing values.
        n (int): The maximum length of gaps to interpolate.

    Returns:
        pd.Series: The Series with small gaps interpolated.
    """
    check_params(param=n, types=int)
    is_nan = series.isna()
    gaps = (is_nan != is_nan.shift()).cumsum()
    mask = series.groupby(gaps).transform("size") <= n
    return series.interpolate().where(mask, series)
