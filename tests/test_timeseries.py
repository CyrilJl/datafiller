import numpy as np
import pandas as pd
import pytest

from datafiller.timeseries import TimeSeriesImputer


@pytest.fixture
def nan_df():
    rng = pd.date_range("2020-01-01", periods=10, freq="D")
    data = {"value": [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10], "value2": [1, 2, 3, 4, 5, 6, 7, 8, np.nan, 10]}
    return pd.DataFrame(data, index=rng)


def test_timeseries_imputer_less_nans(nan_df):
    imputer = TimeSeriesImputer()
    imputed_df = imputer(nan_df)
    assert np.isnan(imputed_df.values).sum() < np.isnan(nan_df.values).sum()


def test_timeseries_imputer_lags(nan_df):
    imputer = TimeSeriesImputer(lags=[1, -1])
    imputed_df = imputer(nan_df)
    assert np.isnan(imputed_df.values).sum() < np.isnan(nan_df.values).sum()


def test_timeseries_imputer_cols_to_impute(nan_df):
    imputer = TimeSeriesImputer()
    imputed_df = imputer(nan_df, cols_to_impute=["value"])
    assert np.isnan(imputed_df["value"]).sum() == 0
    assert np.isnan(imputed_df["value2"]).sum() == np.isnan(nan_df["value2"]).sum()


def test_timeseries_imputer_rows_to_impute(nan_df):
    imputer = TimeSeriesImputer()
    imputed_df = imputer(nan_df, rows_to_impute=nan_df.index[2:7])
    # NaNs outside the range should still be there
    assert np.isnan(imputed_df.loc["2020-01-09", "value2"])
    # NaNs inside the range should be imputed
    assert not np.isnan(imputed_df.loc["2020-01-03", "value"])
    assert not np.isnan(imputed_df.loc["2020-01-07", "value"])


def test_timeseries_imputer_interpolate(nan_df):
    imputer = TimeSeriesImputer(interpolate_gaps_less_than=2)
    imputed_df = imputer(nan_df)
    # The first NaN in 'value' is a gap of 1, so it should be interpolated
    assert not np.isnan(imputed_df.loc["2020-01-03", "value"])
    # The second NaN in 'value' is a gap of 1, so it should be interpolated
    assert not np.isnan(imputed_df.loc["2020-01-07", "value"])
    # The NaN in 'value2' is a gap of 1, so it should be interpolated
    assert not np.isnan(imputed_df.loc["2020-01-09", "value2"])


def test_timeseries_imputer_reindexes_and_imputes_missing_timestamp_block():
    full_index = pd.date_range("2024-01-01", periods=48, freq="h")
    t = np.arange(len(full_index), dtype=np.float32)
    df = pd.DataFrame(
        {
            "load": 0.2 * t + np.sin(2 * np.pi * t / 24),
            "temperature": 10 + np.cos(2 * np.pi * t / 24),
        },
        index=full_index,
    )
    missing_index = full_index[18:24]
    df_missing = df.drop(index=missing_index)

    imputer = TimeSeriesImputer(lags=(1, 2, -1, -2), rng=0)
    imputed_df = imputer(df_missing)

    assert imputed_df.index.equals(full_index)
    assert not imputed_df.loc[missing_index].isna().any().any()
    assert (imputed_df.loc[missing_index] - df.loc[missing_index]).abs().to_numpy().mean() < 0.2


def test_timeseries_imputer_time_features_do_not_duplicate_user_columns():
    full_index = pd.date_range("2024-01-01", periods=36, freq="h")
    t = np.arange(len(full_index), dtype=np.float32)
    df = pd.DataFrame(
        {
            "__time_trend": t,
            "load": np.sin(2 * np.pi * t / 24),
        },
        index=full_index,
    )
    df.loc[full_index[10:14], "load"] = np.nan

    imputed_df = TimeSeriesImputer(lags=(1, -1), rng=0)(df)

    assert list(imputed_df.columns) == ["__time_trend", "load"]
    assert imputed_df.shape == df.shape
    assert not imputed_df.loc[full_index[10:14], "load"].isna().any()


def test_timeseries_imputer_invalid_lags():
    with pytest.raises(ValueError):
        TimeSeriesImputer(lags=[1, 0])


def test_timeseries_imputer_n_nearest_features_tracking(nan_df):
    imputer = TimeSeriesImputer(rng=0, lags=[1, -1])
    n_nearest_features = 3
    imputer(nan_df, n_nearest_features=n_nearest_features)

    assert imputer.imputation_features_ is not None

    cols_with_nans = nan_df.columns[nan_df.isnull().any()].tolist()
    assert set(imputer.imputation_features_.keys()) == set(cols_with_nans)

    for col, features in imputer.imputation_features_.items():
        assert isinstance(col, str)
        assert isinstance(features, list)
        assert all(isinstance(f, str) for f in features)
        assert len(features) <= n_nearest_features
        assert col not in features
        # Check that generated temporal context features are present.
        assert any(("_lag_" in f) or f.startswith("__time_") for f in features)
