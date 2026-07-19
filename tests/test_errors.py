"""Error-path tests: every user-facing validation error should be pinned by a test.

Grouped by the surface that raises: MultivariateImputer configuration, input
validation for arrays and pandas DataFrames, and TimeSeriesImputer validation.
Polars-specific error paths live in the ``*_polars.py`` test modules.
"""

import numpy as np
import pandas as pd
import pytest

from datafiller import (
    DataFillerError,
    DataFillerTypeError,
    DataFillerValueError,
    MultivariateImputer,
    TimeSeriesImputer,
)
from datafiller.multivariate._utils import _validate_input
from datafiller.timeseries._utils import interpolate_small_gaps


@pytest.fixture
def x_valid():
    return np.arange(30, dtype=np.float64).reshape(10, 3)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


def test_exception_hierarchy():
    assert issubclass(DataFillerValueError, DataFillerError)
    assert issubclass(DataFillerValueError, ValueError)
    assert issubclass(DataFillerTypeError, DataFillerError)
    assert issubclass(DataFillerTypeError, TypeError)


def test_validation_errors_are_catchable_as_datafiller_error():
    with pytest.raises(DataFillerError):
        MultivariateImputer()(np.array([1.0, 2.0]))
    with pytest.raises(DataFillerError):
        TimeSeriesImputer()(pd.DataFrame({"a": [1.0, 2.0]}))


# ---------------------------------------------------------------------------
# MultivariateImputer configuration
# ---------------------------------------------------------------------------


def test_multivariate_imputer_invalid_scoring_raises():
    with pytest.raises(ValueError, match="scoring"):
        MultivariateImputer(scoring="mse")


# ---------------------------------------------------------------------------
# MultivariateImputer input validation (numpy)
# ---------------------------------------------------------------------------


def test_validate_input_rejects_non_numpy():
    with pytest.raises(ValueError, match="numpy array"):
        _validate_input([[1.0, 2.0]], None, None, None)


@pytest.mark.parametrize(
    "x, match",
    [
        (np.array([1.0, 2.0, 3.0]), "2D array"),
        (np.array([["a", "b"], ["c", "d"]]), "numeric dtype"),
        (np.array([[1.0, np.inf], [2.0, 3.0]]), "infinity"),
    ],
)
def test_multivariate_imputer_invalid_x_raises(x, match):
    with pytest.raises(ValueError, match=match):
        MultivariateImputer()(x)


@pytest.mark.parametrize(
    "rows, match",
    [
        (np.array([0.5, 1.5]), "integer dtype"),
        (np.array([0, 100]), "between 0 and 9"),
        ([0, 100], "between 0 and 9"),
        (["a"], "between 0 and 9"),
    ],
)
def test_multivariate_imputer_invalid_rows_to_impute_raises(x_valid, rows, match):
    with pytest.raises(ValueError, match=match):
        MultivariateImputer()(x_valid, rows_to_impute=rows)


@pytest.mark.parametrize("cols", [[10], [-1], ["a"]])
def test_multivariate_imputer_invalid_cols_to_impute_raises(x_valid, cols):
    with pytest.raises(ValueError, match="cols_to_impute must be a list of integers between 0 and 2"):
        MultivariateImputer()(x_valid, cols_to_impute=cols)


@pytest.mark.parametrize(
    "n_nearest_features, match",
    [
        (1.5, r"in \(0, 1\]"),
        (0.0, r"in \(0, 1\]"),
        (0.05, "resulted in 0 features"),
        ("two", "int or float"),
        (0, "between 1 and 3"),
        (10, "between 1 and 3"),
    ],
)
def test_multivariate_imputer_invalid_n_nearest_features_raises(x_valid, n_nearest_features, match):
    with pytest.raises(ValueError, match=match):
        MultivariateImputer()(x_valid, n_nearest_features=n_nearest_features)


# ---------------------------------------------------------------------------
# MultivariateImputer input validation (pandas)
# ---------------------------------------------------------------------------


def test_multivariate_imputer_unknown_row_label_raises():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=["r0", "r1"])
    with pytest.raises(ValueError, match=r"Row labels not found in index: \['r2'\]"):
        MultivariateImputer()(df, rows_to_impute=["r2"])


def test_multivariate_imputer_unknown_column_label_raises():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(ValueError, match=r"Column labels not found in columns: \['c'\]"):
        MultivariateImputer()(df, cols_to_impute=["c"])


# ---------------------------------------------------------------------------
# TimeSeriesImputer configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lags", [1, ["a"], [1.5]])
def test_timeseries_imputer_lags_must_be_iterable_of_ints(lags):
    with pytest.raises(ValueError, match="lags must be an iterable of integers"):
        TimeSeriesImputer(lags=lags)


def test_timeseries_imputer_time_column_must_be_string_or_none():
    with pytest.raises(ValueError, match="string or None"):
        TimeSeriesImputer(time_column=1)

    imputer = TimeSeriesImputer()
    with pytest.raises(ValueError, match="string or None"):
        imputer.set_params(time_column=1)


# ---------------------------------------------------------------------------
# TimeSeriesImputer input validation
# ---------------------------------------------------------------------------


def _ts_frame(periods=30):
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    values = np.arange(periods, dtype=np.float64)
    values[3] = np.nan
    return pd.DataFrame({"value": values, "other": values + 1.0}, index=index)


def test_timeseries_imputer_rejects_non_dataframe():
    with pytest.raises(TypeError, match="pandas or eager Polars"):
        TimeSeriesImputer()(np.zeros((5, 2)))


def test_timeseries_imputer_rejects_non_datetime_index():
    with pytest.raises(TypeError, match="DatetimeIndex"):
        TimeSeriesImputer()(pd.DataFrame({"a": [1.0, 2.0]}))


@pytest.mark.parametrize(
    "timestamps, match",
    [
        (["2024-01-01"], "at least two timestamps"),
        (["2024-01-02", "2024-01-01"], "sorted in increasing order"),
        (["2024-01-01", "2024-01-01"], "duplicate timestamps"),
        (["2024-01-01", "2024-01-02", "2024-01-03 12:00"], "irregular timestamp gaps"),
    ],
)
def test_timeseries_imputer_invalid_index_raises(timestamps, match):
    index = pd.DatetimeIndex(timestamps)
    df = pd.DataFrame({"a": np.ones(len(index))}, index=index)
    with pytest.raises(ValueError, match=match):
        TimeSeriesImputer()(df)


def test_timeseries_imputer_invalid_cols_to_impute_type_raises():
    with pytest.raises(ValueError, match="int, str, or an iterable"):
        TimeSeriesImputer()(_ts_frame(), cols_to_impute=[1.5])


def test_timeseries_imputer_rejects_unsupported_dtype_columns():
    # Categorical/string/bool columns are supported; datetime data columns are not.
    df = _ts_frame()
    df["when"] = df.index
    with pytest.raises(ValueError, match="requires numeric columns"):
        TimeSeriesImputer()(df)


def test_interpolate_small_gaps_requires_int():
    series = pd.Series([1.0, np.nan, 3.0])
    with pytest.raises(TypeError, match="n must be an int"):
        interpolate_small_gaps(series, 1.5)
