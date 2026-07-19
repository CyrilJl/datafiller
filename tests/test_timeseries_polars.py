from datetime import date, datetime, timedelta

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from datafiller import TimeSeriesImputer

pl = pytest.importorskip("polars")


def _polars_time_series(periods=24):
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(periods)]
    values = np.arange(periods, dtype=float)
    values[5] = np.nan
    return pl.DataFrame({"timestamp": timestamps, "value": values, "other": values + 1})


def test_timeseries_imputer_polars_returns_polars_and_preserves_schema():
    df = _polars_time_series()

    imputed = TimeSeriesImputer(time_column="timestamp", lags=(1, -1), rng=0)(df)

    assert isinstance(imputed, pl.DataFrame)
    assert imputed.columns == df.columns
    assert imputed.schema == df.schema
    assert imputed.select(pl.col("value").is_nan().sum()).item() == 0


def test_timeseries_imputer_polars_reinserts_missing_timestamp_block():
    full = _polars_time_series(48)
    missing_timestamps = full["timestamp"][18:24].to_list()
    incomplete = full.filter(~pl.col("timestamp").is_in(missing_timestamps))

    imputed = TimeSeriesImputer(time_column="timestamp", lags=(1, 2, -1, -2), rng=0)(incomplete)

    assert imputed.height == full.height
    assert imputed["timestamp"].to_list() == full["timestamp"].to_list()
    assert imputed.filter(pl.col("timestamp").is_in(missing_timestamps))["value"].null_count() == 0


def test_timeseries_imputer_polars_selectors_accept_positions_and_timestamps():
    df = _polars_time_series(12).with_columns(
        pl.when(pl.int_range(pl.len()) == 8).then(None).otherwise("other").alias("other")
    )
    target_timestamp = df["timestamp"][5]
    imputer = TimeSeriesImputer(time_column="timestamp", min_samples_train=20)

    by_timestamp = imputer(df, rows_to_impute=[target_timestamp], cols_to_impute=["value"])
    by_position = imputer(df, rows_to_impute=[5], cols_to_impute=["value"])

    assert not np.isnan(by_timestamp["value"][5])
    assert not np.isnan(by_position["value"][5])
    assert by_timestamp["other"].null_count() == df["other"].null_count()


def test_timeseries_imputer_polars_pipeline_fit_transform():
    df = _polars_time_series()
    pipeline = Pipeline([("imputer", TimeSeriesImputer(time_column="timestamp"))])

    imputed = pipeline.fit_transform(df)

    assert isinstance(imputed, pl.DataFrame)
    assert imputed.shape == df.shape


def test_timeseries_imputer_polars_validates_time_column():
    df = _polars_time_series()
    with pytest.raises(ValueError, match="string or None"):
        TimeSeriesImputer(time_column=1)
    with pytest.raises(ValueError, match="time_column must be set"):
        TimeSeriesImputer()(df)
    with pytest.raises(ValueError, match="was not found"):
        TimeSeriesImputer(time_column="missing")(df)
    with pytest.raises(TypeError, match="Date or Datetime"):
        TimeSeriesImputer(time_column="value")(df)
    with pytest.raises(TypeError, match="collect"):
        TimeSeriesImputer(time_column="timestamp")(df.lazy())


def test_timeseries_imputer_polars_preserves_date_column_dtype():
    dates = pl.Series("date", [date(2024, 1, 1) + timedelta(days=i) for i in [0, 1, 3, 4]], dtype=pl.Date)
    df = pl.DataFrame({"date": dates, "value": [1.0, 2.0, 4.0, 5.0]})

    imputed = TimeSeriesImputer(time_column="date", lags=(1, -1))(df)

    assert imputed.schema["date"] == pl.Date
    assert imputed.height == 5
