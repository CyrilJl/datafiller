"""Tests for the optional GPU (PyTorch) backend of MultivariateImputer.

The whole module is skipped when torch is not installed; tests that need a
GPU are additionally skipped when no CUDA device is available.
"""

import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="GPU tests require the optional torch dependency")

from datafiller import MultivariateImputer, TimeSeriesImputer  # noqa: E402
from datafiller.estimators.ridge import FastRidge  # noqa: E402
from datafiller.multivariate import _gpu  # noqa: E402

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")


def make_correlated_matrix(m=400, n=25, nan_ratio=0.05, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(m, 5))
    x = latent @ rng.normal(size=(5, n)) + 0.05 * rng.normal(size=(m, n))
    x = x.astype(dtype)
    x[rng.random((m, n)) < nan_ratio] = np.nan
    return x


def assert_same_imputation(gpu_out, cpu_out, atol=1e-2):
    np.testing.assert_array_equal(np.isnan(gpu_out), np.isnan(cpu_out))
    np.testing.assert_allclose(gpu_out, cpu_out, rtol=0, atol=atol)


@requires_cuda
def test_gpu_matches_cpu_all_features():
    x = make_correlated_matrix()
    out_cpu = MultivariateImputer(rng=0)(x)
    out_gpu = MultivariateImputer(rng=0, device="cuda")(x)
    assert not np.isnan(out_gpu).any()
    assert_same_imputation(out_gpu, out_cpu)


@requires_cuda
def test_gpu_matches_cpu_with_feature_sampling():
    x = make_correlated_matrix(m=600, n=40)
    out_cpu = MultivariateImputer(rng=0)(x, n_nearest_features=10)
    out_gpu = MultivariateImputer(rng=0, device="cuda")(x, n_nearest_features=10)
    assert_same_imputation(out_gpu, out_cpu)


@requires_cuda
def test_gpu_matches_cpu_float64_input():
    x = make_correlated_matrix(dtype=np.float64)
    out_cpu = MultivariateImputer(rng=0)(x)
    out_gpu = MultivariateImputer(rng=0, device="cuda")(x)
    assert_same_imputation(out_gpu, out_cpu)


@requires_cuda
def test_gpu_matches_cpu_without_intercept():
    x = make_correlated_matrix()
    regressor = FastRidge(fit_intercept=False)
    out_cpu = MultivariateImputer(rng=0, regressor=regressor)(x)
    out_gpu = MultivariateImputer(rng=0, regressor=FastRidge(fit_intercept=False), device="cuda")(x)
    assert_same_imputation(out_gpu, out_cpu)


@requires_cuda
def test_gpu_falls_back_to_optimask_for_sparse_patterns():
    # Heavy missingness + a high min_samples_train forces some patterns below
    # the complete-row threshold, exercising the CPU optimask fallback.
    x = make_correlated_matrix(m=80, n=8, nan_ratio=0.3, seed=3)
    out_cpu = MultivariateImputer(rng=0, min_samples_train=30)(x)
    out_gpu = MultivariateImputer(rng=0, min_samples_train=30, device="cuda")(x)
    assert_same_imputation(out_gpu, out_cpu)


@requires_cuda
def test_gpu_matches_cpu_row_chunked_grams(monkeypatch):
    # Shrink the chunking budgets so the accumulation loops are exercised on
    # a small matrix.
    monkeypatch.setattr(_gpu.GramBackend, "_OUTER_BUDGET_BYTES", 64_000)
    monkeypatch.setattr(_gpu.GramBackend, "_WEIGHT_BUDGET_BYTES", 16_000)
    x = make_correlated_matrix(m=500, n=12)
    out_cpu = MultivariateImputer(rng=0)(x)
    out_gpu = MultivariateImputer(rng=0, device="cuda")(x)
    assert_same_imputation(out_gpu, out_cpu)


@requires_cuda
def test_gpu_dataframe_with_categorical_column():
    x = make_correlated_matrix(m=200, n=4, seed=1)
    rng = np.random.default_rng(1)
    labels = np.where(x[:, 0] > np.nanmedian(x[:, 0]), "high", "low").astype(object)
    labels[rng.random(len(labels)) < 0.1] = np.nan
    df = pd.DataFrame(x, columns=[f"num_{i}" for i in range(4)])
    df["cat"] = labels

    out_cpu = MultivariateImputer(rng=0)(df)
    out_gpu = MultivariateImputer(rng=0, device="cuda")(df)
    assert not out_gpu.isna().any().any()
    pd.testing.assert_series_equal(out_gpu["cat"], out_cpu["cat"])
    assert_same_imputation(
        out_gpu.drop(columns="cat").to_numpy(np.float64),
        out_cpu.drop(columns="cat").to_numpy(np.float64),
    )


@requires_cuda
def test_timeseries_device_matches_cpu():
    index = pd.date_range("2024-01-01", periods=500, freq="h")
    rng = np.random.default_rng(0)
    base = np.sin(np.arange(500) / 10)[:, None] + 0.1 * rng.normal(size=(500, 5))
    df = pd.DataFrame(base.astype(np.float32), index=index, columns=list("abcde"))
    df.iloc[100:130, 2] = np.nan
    df[rng.random(df.shape) < 0.03] = np.nan

    out_cpu = TimeSeriesImputer(lags=(1, -1), rng=0)(df)
    out_gpu = TimeSeriesImputer(lags=(1, -1), rng=0, device="cuda")(df)
    assert_same_imputation(out_gpu.to_numpy(np.float64), out_cpu.to_numpy(np.float64))


def test_invalid_device_raises():
    x = make_correlated_matrix(m=50, n=5)
    with pytest.raises(RuntimeError):
        MultivariateImputer(device="not-a-device")(x)


def test_missing_torch_raises_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    x = make_correlated_matrix(m=50, n=5)
    with pytest.raises(ImportError, match=r"datafiller\[gpu\]"):
        MultivariateImputer(device="cuda")(x)


def test_device_none_never_builds_backend():
    x = make_correlated_matrix(m=50, n=5)
    imputer = MultivariateImputer(rng=0)
    imputer(x)
    assert imputer._gpu_backend is None
