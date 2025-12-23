import pandas as pd


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
        raise TypeError("n must be an int")
    is_nan = series.isna()
    gaps = (is_nan != is_nan.shift()).cumsum()
    mask = series.groupby(gaps).transform("size") <= n
    return series.interpolate().where(mask, series)
