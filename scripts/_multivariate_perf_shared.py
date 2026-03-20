"""Shared helpers for multivariate imputer benchmarks and profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype


@dataclass(frozen=True)
class BenchmarkPayload:
    """A fully materialized benchmark input and its ground truth."""

    name: str
    truth: np.ndarray | pd.DataFrame
    masked: np.ndarray | pd.DataFrame
    mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkCase:
    """A registered benchmark configuration."""

    name: str
    description: str
    dataset_kind: str
    builder: Callable[[int], BenchmarkPayload]
    call_kwargs: dict[str, Any] = field(default_factory=dict)


def is_categorical_series(series: pd.Series) -> bool:
    """Return True when a series should be scored as categorical."""
    return bool(
        is_object_dtype(series.dtype)
        or is_string_dtype(series.dtype)
        or is_bool_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def clone_input_data(data: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """Clone benchmark input so each timed run gets a fresh object."""
    if isinstance(data, pd.DataFrame):
        return data.copy(deep=True)
    return data.copy()


def make_mar_mask(shape: tuple[int, int], missing_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a missing-at-random mask."""
    return rng.random(shape) < missing_ratio


def make_block_mask(
    shape: tuple[int, int],
    frac_columns: float,
    block_length_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate contiguous missing blocks across a subset of columns."""
    n_rows, n_cols = shape
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    n_cols_to_mask = max(1, int(n_cols * frac_columns))
    cols = rng.choice(np.arange(n_cols), size=n_cols_to_mask, replace=False)
    block_length = max(1, int(n_rows * block_length_ratio))
    for col in cols:
        start = rng.integers(0, max(1, n_rows - block_length + 1))
        mask[start : start + block_length, col] = True
    return mask


def _make_correlated_numeric_array(seed: int, rows: int, cols: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean = np.linspace(-0.75, 0.75, cols)
    cov = np.fromfunction(lambda i, j: 0.82 ** np.abs(i - j), (cols, cols), dtype=float)
    x = rng.multivariate_normal(mean=mean, cov=cov, size=rows).astype(np.float32)

    # Add a few deterministic interactions so the regression target is not purely Gaussian.
    if cols >= 6:
        x[:, 0] = (0.35 * x[:, 1] - 0.20 * x[:, 2] + 0.10 * x[:, 3] ** 2).astype(np.float32)
        x[:, 4] = (x[:, 4] + 0.50 * x[:, 5]).astype(np.float32)
    return x


def build_numeric_payload(
    *,
    seed: int,
    name: str,
    rows: int,
    cols: int,
    mask_kind: str,
    missing_ratio: float | None = None,
    frac_columns: float | None = None,
    block_length_ratio: float | None = None,
) -> BenchmarkPayload:
    """Build a numeric array benchmark payload."""
    rng = np.random.default_rng(seed)
    truth = _make_correlated_numeric_array(seed=seed, rows=rows, cols=cols)
    if mask_kind == "mar":
        if missing_ratio is None:
            raise ValueError("missing_ratio is required for MAR masks.")
        mask = make_mar_mask(truth.shape, missing_ratio, rng)
    elif mask_kind == "block":
        if frac_columns is None or block_length_ratio is None:
            raise ValueError("frac_columns and block_length_ratio are required for block masks.")
        mask = make_block_mask(truth.shape, frac_columns, block_length_ratio, rng)
    else:
        raise ValueError(f"Unknown mask_kind: {mask_kind}")

    masked = truth.copy()
    masked[mask] = np.nan
    return BenchmarkPayload(
        name=name,
        truth=truth,
        masked=masked,
        mask=mask,
        metadata={
            "rows": rows,
            "cols": cols,
            "missing_ratio": float(mask.mean()),
            "masked_total": int(mask.sum()),
            "dataset_kind": "array",
            "mask_kind": mask_kind,
        },
    )


def build_mixed_dataframe_payload(
    *,
    seed: int,
    name: str,
    rows: int,
    missing_ratio: float,
) -> BenchmarkPayload:
    """Build a mixed numeric/categorical dataframe benchmark payload."""
    rng = np.random.default_rng(seed)

    base_age = rng.normal(42, 11, size=rows)
    tenure = rng.gamma(shape=2.2, scale=3.5, size=rows)
    income = np.exp(9.5 + 0.035 * base_age + 0.06 * tenure + rng.normal(0, 0.22, size=rows))
    score = 520 + 3.8 * base_age + 1.5 * tenure + rng.normal(0, 18, size=rows)
    debt_ratio = np.clip((income / income.mean()) * 0.16 + rng.normal(0.0, 0.03, size=rows), 0.01, 0.95)

    segment = np.where(score > 710, "enterprise", np.where(score > 640, "growth", "starter"))
    region = np.where(base_age < 35, "north", np.where(base_age < 45, "west", np.where(base_age < 55, "east", "south")))
    channel = np.where(tenure > np.median(tenure), "direct", "partner")
    is_active = pd.Series((score + rng.normal(0, 10, size=rows)) > np.median(score), dtype="boolean")

    truth = pd.DataFrame(
        {
            "age": base_age.round(1),
            "tenure_years": tenure.round(2),
            "income": income.round(2),
            "score": score.round(1),
            "debt_ratio": debt_ratio.round(4),
            "segment": pd.Categorical(segment, categories=["starter", "growth", "enterprise"]),
            "region": pd.Categorical(region, categories=["north", "west", "east", "south"]),
            "channel": pd.Series(channel, dtype="string"),
            "is_active": is_active,
        }
    )

    mask = make_mar_mask(truth.shape, missing_ratio, rng)
    masked = truth.copy(deep=True)
    for col_idx, col in enumerate(masked.columns):
        col_mask = mask[:, col_idx]
        if is_bool_dtype(masked[col].dtype):
            masked.loc[col_mask, col] = pd.NA
        else:
            masked.loc[col_mask, col] = np.nan

    return BenchmarkPayload(
        name=name,
        truth=truth,
        masked=masked,
        mask=mask,
        metadata={
            "rows": rows,
            "cols": truth.shape[1],
            "missing_ratio": float(mask.mean()),
            "masked_total": int(mask.sum()),
            "dataset_kind": "dataframe",
            "mask_kind": "mar",
            "numeric_cols": int(sum(not is_categorical_series(truth[col]) for col in truth.columns)),
            "categorical_cols": int(sum(is_categorical_series(truth[col]) for col in truth.columns)),
        },
    )


def evaluate_imputation(
    truth: np.ndarray | pd.DataFrame,
    imputed: np.ndarray | pd.DataFrame,
    mask: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate a benchmark run on the originally masked entries."""
    if isinstance(truth, np.ndarray):
        y_true = truth[mask]
        y_pred = np.asarray(imputed)[mask]
        finite = np.isfinite(y_pred)
        errors = y_pred[finite] - y_true[finite]
        return {
            "rmse": float(np.sqrt(np.mean(errors**2))) if errors.size else np.nan,
            "mae": float(np.mean(np.abs(errors))) if errors.size else np.nan,
            "accuracy": np.nan,
            "coverage": float(finite.mean()) if finite.size else np.nan,
            "remaining_missing_total": int(np.isnan(np.asarray(imputed)).sum()),
        }

    numeric_true: list[np.ndarray] = []
    numeric_pred: list[np.ndarray] = []
    categorical_matches: list[bool] = []
    categorical_valid: list[bool] = []
    remaining_missing_total = 0

    mask_df = pd.DataFrame(mask, columns=truth.columns, index=truth.index)
    for col in truth.columns:
        col_mask = mask_df[col].to_numpy()
        if not np.any(col_mask):
            continue

        imputed_series = imputed.loc[col_mask, col]
        remaining_missing_total += int(imputed_series.isna().sum())

        if is_categorical_series(truth[col]):
            y_true = truth.loc[col_mask, col].reset_index(drop=True)
            y_pred = imputed.loc[col_mask, col].reset_index(drop=True)
            valid = ~(y_true.isna() | y_pred.isna())
            categorical_valid.extend(valid.tolist())
            categorical_matches.extend((y_true[valid] == y_pred[valid]).tolist())
        else:
            y_true = truth.loc[col_mask, col].to_numpy(dtype=float)
            y_pred = imputed.loc[col_mask, col].to_numpy(dtype=float)
            finite = np.isfinite(y_true) & np.isfinite(y_pred)
            if np.any(finite):
                numeric_true.append(y_true[finite])
                numeric_pred.append(y_pred[finite])

    if numeric_true:
        numeric_true_arr = np.concatenate(numeric_true)
        numeric_pred_arr = np.concatenate(numeric_pred)
        errors = numeric_pred_arr - numeric_true_arr
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        numeric_coverage = float(len(numeric_pred_arr) / int(np.sum(mask[:, [not is_categorical_series(truth[c]) for c in truth.columns]])))
    else:
        rmse = np.nan
        mae = np.nan
        numeric_coverage = np.nan

    if categorical_valid:
        valid_total = sum(categorical_valid)
        accuracy = float(sum(categorical_matches) / valid_total) if valid_total else np.nan
        categorical_coverage = float(valid_total / len(categorical_valid)) if categorical_valid else np.nan
    else:
        accuracy = np.nan
        categorical_coverage = np.nan

    coverage_values = [value for value in [numeric_coverage, categorical_coverage] if not np.isnan(value)]
    coverage = float(np.mean(coverage_values)) if coverage_values else np.nan
    return {
        "rmse": rmse,
        "mae": mae,
        "accuracy": accuracy,
        "coverage": coverage,
        "remaining_missing_total": remaining_missing_total,
    }
