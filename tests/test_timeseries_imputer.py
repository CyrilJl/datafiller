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
