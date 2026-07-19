"""Tests for the synthetic missingness helpers in datafiller.datasets."""

import numpy as np
import pandas as pd
import pytest

from datafiller.datasets import add_contiguous_missing, add_mar


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(200, 8)), columns=[f"col_{i}" for i in range(8)])


@pytest.mark.parametrize("nan_ratio", [-0.1, 1.5])
def test_add_mar_invalid_ratio_raises(df, nan_ratio):
    with pytest.raises(ValueError, match="nan_ratio must be between 0 and 1"):
        add_mar(df, nan_ratio=nan_ratio)


def test_add_mar_hits_requested_ratio(df):
    result = add_mar(df, nan_ratio=0.3, rng=0)
    observed_ratio = result.isna().to_numpy().mean()
    assert observed_ratio == pytest.approx(0.3, abs=0.05)
    assert not df.isna().any().any(), "input DataFrame must not be modified"


@pytest.mark.parametrize("nan_ratio, expected", [(0.0, 0.0), (1.0, 1.0)])
def test_add_mar_boundary_ratios(df, nan_ratio, expected):
    result = add_mar(df, nan_ratio=nan_ratio, rng=0)
    assert result.isna().to_numpy().mean() == expected


def test_add_mar_reproducible_with_seed(df):
    first = add_mar(df, nan_ratio=0.2, rng=42)
    second = add_mar(df, nan_ratio=0.2, rng=42)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("frac_columns", [-0.1, 1.5])
def test_add_contiguous_missing_invalid_frac_raises(df, frac_columns):
    with pytest.raises(ValueError, match="frac_columns must be between 0 and 1"):
        add_contiguous_missing(df, frac_columns=frac_columns, length=10)


def test_add_contiguous_missing_int_length(df):
    result = add_contiguous_missing(df, frac_columns=0.5, length=10, rng=0)

    modified = [col for col in df.columns if result[col].isna().any()]
    assert len(modified) == 4  # half of the 8 columns
    assert not df.isna().any().any(), "input DataFrame must not be modified"
    for col in modified:
        nan_positions = np.flatnonzero(result[col].isna().to_numpy())
        assert len(nan_positions) == 10
        assert np.array_equal(nan_positions, np.arange(nan_positions[0], nan_positions[0] + 10)), (
            f"missing block in {col} is not contiguous"
        )


def test_add_contiguous_missing_float_length(df):
    result = add_contiguous_missing(df, frac_columns=0.25, length=0.1, rng=0)

    modified = [col for col in df.columns if result[col].isna().any()]
    assert len(modified) == 2
    for col in modified:
        assert result[col].isna().sum() == 20  # 10% of 200 rows


def test_add_contiguous_missing_length_capped_at_n_rows(df):
    result = add_contiguous_missing(df, frac_columns=1.0, length=10_000, rng=0)
    assert result.isna().all().all()


def test_add_contiguous_missing_reproducible_with_seed(df):
    first = add_contiguous_missing(df, frac_columns=0.5, length=15, rng=7)
    second = add_contiguous_missing(df, frac_columns=0.5, length=15, rng=7)
    pd.testing.assert_frame_equal(first, second)
