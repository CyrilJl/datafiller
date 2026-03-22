import time

import numpy as np
import pandas as pd

from datafiller import MultivariateImputer, TimeSeriesImputer


def _make_tsi_panel(
    n_rows: int = 30_000,
    n_series: int = 250,
    n_latent_factors: int = 12,
    seed: int = 0,
    freq: str = "15min",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    innovations = rng.normal(scale=0.3, size=(n_rows, n_latent_factors)).astype(np.float32)
    latent = np.empty_like(innovations)
    latent[0] = innovations[0]
    for row in range(1, n_rows):
        latent[row] = 0.93 * latent[row - 1] + innovations[row]

    t = np.arange(n_rows, dtype=np.float32)
    seasonal = np.column_stack(
        [
            np.sin(2 * np.pi * t / 24),
            np.cos(2 * np.pi * t / 24),
            np.sin(2 * np.pi * t / (24 * 7)),
            np.cos(2 * np.pi * t / (24 * 7)),
        ]
    ).astype(np.float32)
    factors = np.concatenate([latent, seasonal], axis=1)

    loadings = rng.normal(scale=0.35, size=(factors.shape[1], n_series)).astype(np.float32)
    n_groups = min(n_latent_factors, 12)
    for col in range(n_series):
        group = col % n_groups
        loadings[group, col] += 1.5
        loadings[n_latent_factors + (group % seasonal.shape[1]), col] += 0.4

    data = factors @ loadings
    data += rng.normal(scale=0.08, size=data.shape).astype(np.float32)
    data += np.linspace(-0.5, 0.5, n_rows, dtype=np.float32)[:, None] * rng.normal(
        scale=0.15,
        size=(1, n_series),
    ).astype(np.float32)

    index = pd.date_range("2024-01-01", periods=n_rows, freq=freq)
    columns = [f"series_{i:03d}" for i in range(n_series)]
    return pd.DataFrame(data, index=index, columns=columns)


def _apply_tsi_missingness(
    df: pd.DataFrame,
    n_target_series: int = 3,
    n_block_series: int = 8,
    mar_ratio: float = 0.02,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed + 1)
    values = df.to_numpy(dtype=np.float32, copy=True)

    mar_mask = rng.random(values.shape) < mar_ratio
    values[mar_mask] = np.nan

    block_indices = rng.choice(df.shape[1], size=n_block_series, replace=False)
    target_indices = np.sort(block_indices[:n_target_series])

    block_length = max(24, len(df) // 10)
    max_start = len(df) - block_length
    for col_idx in block_indices:
        start = int(rng.integers(0, max_start + 1))
        end = start + block_length
        values[start:end, col_idx] = np.nan

    target_columns = df.columns[target_indices].tolist()
    return pd.DataFrame(values, index=df.index, columns=df.columns), target_columns


def test_multivariate_imputer_timing():
    rng = np.random.RandomState(0)
    x = rng.normal(size=(25_000, 25)).astype(np.float32)
    missing_mask = rng.rand(*x.shape) < 0.05
    x[missing_mask] = np.nan

    imputer = MultivariateImputer(verbose=0)

    # Warm up JIT compilation.
    _ = imputer(x)

    start = time.perf_counter()
    _ = imputer(x)
    elapsed = time.perf_counter() - start

    print(f"MultivariateImputer elapsed seconds: {elapsed:.2f}s")


def test_tsi_timing():
    df = _make_tsi_panel()
    df_missing, target_columns = _apply_tsi_missingness(df)

    imputer = TimeSeriesImputer(lags=(1, 2, 3, -1, -2, -3), rng=0)

    start = time.perf_counter()
    _ = imputer(df_missing, cols_to_impute=target_columns, n_nearest_features=35)
    elapsed = time.perf_counter() - start

    print(f"TimeSeriesImputer elapsed seconds: {elapsed:.2f}s")
