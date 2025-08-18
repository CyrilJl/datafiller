import numpy as np
import pandas as pd
import pytest
from datafiller import TimeSeriesImputer


def generate_ts_data(n_samples, n_features):
    dates = pd.to_datetime(pd.date_range(start="2020-01-01", periods=n_samples, freq="D"))
    data = np.random.rand(n_samples, n_features)
    return pd.DataFrame(data, index=dates, columns=[f"feature_{i}" for i in range(n_features)])


def test_timeseries_imputer_smoke():
    df = generate_ts_data(100, 3)
    df.iloc[10:20, 0] = np.nan
    df.iloc[30:40, 1] = np.nan

    ts_imputer = TimeSeriesImputer(lags=[1, 2])
    imputed_df = ts_imputer(df)

    assert not imputed_df.isnull().sum().sum()


def test_timeseries_imputer_col_subset():
    df = generate_ts_data(100, 3)
    df.iloc[10:20, 0] = np.nan
    df.iloc[30:40, 1] = np.nan
    df.iloc[50:60, 2] = np.nan

    ts_imputer = TimeSeriesImputer(lags=[1, 2])
    imputed_df = ts_imputer(df, cols_to_impute=["feature_0", "feature_2"])

    assert not imputed_df["feature_0"].isnull().sum()
    assert imputed_df["feature_1"].isnull().sum() > 0
    assert not imputed_df["feature_2"].isnull().sum()


def test_timeseries_imputer_negative_lags():
    df = generate_ts_data(100, 2)
    df.iloc[10:20, 0] = np.nan
    df.iloc[30:40, 1] = np.nan

    ts_imputer = TimeSeriesImputer(lags=[-1, 1, 2])
    imputed_df = ts_imputer(df)

    assert not imputed_df.isnull().sum().sum()


def test_timeseries_imputer_invalid_input():
    with pytest.raises(TypeError):
        ts_imputer = TimeSeriesImputer()
        ts_imputer(np.random.rand(10, 10))

    df = generate_ts_data(100, 2)
    df_no_freq = df.copy()
    df_no_freq.index = pd.DatetimeIndex(df_no_freq.index.values)
    with pytest.raises(ValueError):
        ts_imputer = TimeSeriesImputer()
        ts_imputer(df_no_freq)

    with pytest.raises(ValueError):
        TimeSeriesImputer(lags=[1, 0])


def test_timeseries_imputer_before():
    df = generate_ts_data(100, 2)
    df.iloc[10:20, 0] = np.nan
    df.iloc[80:90, 1] = np.nan

    ts_imputer = TimeSeriesImputer(lags=[1, 2])
    imputed_df = ts_imputer(df, before="2020-02-19")  # 50th day

    assert not imputed_df.iloc[:50]["feature_0"].isnull().sum()
    assert imputed_df.iloc[50:]["feature_0"].isnull().sum() == 0
    assert not imputed_df.iloc[:50]["feature_1"].isnull().sum()
    assert imputed_df.iloc[50:]["feature_1"].isnull().sum() > 0


def test_timeseries_imputer_after():
    df = generate_ts_data(100, 2)
    df.iloc[10:20, 0] = np.nan
    df.iloc[80:90, 1] = np.nan

    ts_imputer = TimeSeriesImputer(lags=[1, 2])
    imputed_df = ts_imputer(df, after="2020-02-19")  # 50th day

    assert imputed_df.iloc[:50]["feature_0"].isnull().sum() > 0
    assert not imputed_df.iloc[50:]["feature_0"].isnull().sum()
    assert imputed_df.iloc[:50]["feature_1"].isnull().sum() == 0
    assert not imputed_df.iloc[50:]["feature_1"].isnull().sum()


def test_timeseries_imputer_before_and_after():
    df = generate_ts_data(100, 2)
    df.iloc[10:20, 0] = np.nan
    df.iloc[40:50, 0] = np.nan
    df.iloc[80:90, 1] = np.nan

    ts_imputer = TimeSeriesImputer(lags=[1, 2])
    imputed_df = ts_imputer(df, after="2020-01-30", before="2020-02-29")  # 30th and 60th day

    assert imputed_df.iloc[:30]["feature_0"].isnull().sum() > 0
    assert not imputed_df.iloc[30:60]["feature_0"].isnull().sum()
    assert imputed_df.iloc[60:]["feature_1"].isnull().sum() > 0


def test_timeseries_imputer_rows_to_impute_priority():
    df = generate_ts_data(100, 2)
    df.iloc[10:20, 0] = np.nan
    df.iloc[80:90, 1] = np.nan

    ts_imputer = TimeSeriesImputer(lags=[1, 2])
    imputed_df = ts_imputer(df, rows_to_impute=range(10, 20), before="2020-01-50")

    assert not imputed_df.iloc[10:20]["feature_0"].isnull().sum()
    assert imputed_df.iloc[80:90]["feature_1"].isnull().sum() > 0


def test_timeseries_imputer_invalid_lags():
    with pytest.raises(ValueError, match="lags must be an iterable of integers"):
        TimeSeriesImputer(lags=1)
    with pytest.raises(ValueError, match="lags must be an iterable of integers"):
        TimeSeriesImputer(lags=[1, "a"])


def test_timeseries_imputer_invalid_index():
    df = pd.DataFrame(np.random.rand(10, 2))
    ts_imputer = TimeSeriesImputer()
    with pytest.raises(TypeError, match="DataFrame index must be a DatetimeIndex"):
        ts_imputer(df)


def test_timeseries_imputer_single_col_to_impute():
    df = generate_ts_data(100, 3)
    df.iloc[10:20, 0] = np.nan
    df.iloc[30:40, 1] = np.nan
    ts_imputer = TimeSeriesImputer(lags=[1, 2])

    # Test with single string
    imputed_df_str = ts_imputer(df.copy(), cols_to_impute="feature_0")
    assert not imputed_df_str["feature_0"].isnull().sum()
    assert imputed_df_str["feature_1"].isnull().sum() > 0

    # Test with single int
    imputed_df_int = ts_imputer(df.copy(), cols_to_impute=0)
    assert not imputed_df_int["feature_0"].isnull().sum()
    assert imputed_df_int["feature_1"].isnull().sum() > 0


def test_timeseries_imputer_invalid_col_type():
    df = generate_ts_data(100, 2)
    ts_imputer = TimeSeriesImputer()
    with pytest.raises(ValueError, match="cols_to_impute must be an int, str, or an iterable of ints or strs"):
        ts_imputer(df, cols_to_impute=[0, 1.5])


def test_interpolate_small_gaps():
    df = generate_ts_data(100, 2)
    # Small gap of 2 in feature 0
    df.iloc[10:12, 0] = np.nan
    # Large gap of 10 in feature 0
    df.iloc[20:30, 0] = np.nan
    # Some NaNs in feature 1, but not overlapping with the large gap in feature 0
    df.iloc[5:7, 1] = np.nan
    df.iloc[40:45, 1] = np.nan

    ts_imputer = TimeSeriesImputer(interpolate_gaps_less_than=3, lags=[1])
    imputed_df = ts_imputer(df)

    # Small gap should be imputed by interpolation
    assert not imputed_df.iloc[10:12, 0].isnull().any()

    # Large gap should be imputed by the multivariate imputer
    assert not imputed_df.iloc[20:30, 0].isnull().any()

    # All NaNs should be imputed
    assert not imputed_df.isnull().sum().sum()
