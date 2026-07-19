import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)

from datafiller import MultivariateImputer, TimeSeriesImputer

SEED = 0

TS_LAGS = (1, 2, 3, -1, -2, -3)


def fetch_file(url: str, known_hash: str) -> str:
    import pooch

    return pooch.retrieve(url=url, known_hash=known_hash, progressbar=True)


def load_synthetic_mixed_df(seed: int, n_rows: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.normal(40, 12, size=n_rows).round(1),
            "income": rng.lognormal(mean=10.2, sigma=0.45, size=n_rows),
            "score": rng.normal(620, 55, size=n_rows).round(0),
            "segment": pd.Categorical(rng.choice(["basic", "plus", "pro"], size=n_rows, p=[0.5, 0.35, 0.15])),
            "region": pd.Categorical(rng.choice(["north", "south", "east", "west"], size=n_rows)),
            "is_active": rng.choice([True, False], size=n_rows, p=[0.7, 0.3]),
        }
    )
    return df


def load_spambase() -> pd.DataFrame:
    """Spambase (UCI): 57 numeric features; used in GAIN/MIWAE imputation papers."""
    path = fetch_file(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
        "md5:e7368d48fa690b0331a09b430d55fb6a",
    )
    df = pd.read_csv(path, header=None)
    return df.iloc[:, :-1].rename(columns=lambda i: f"f{i}")


def load_letter_recognition() -> pd.DataFrame:
    """Letter Recognition (UCI): 16 numeric features + 26-class letter; used in GAIN."""
    path = fetch_file(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
        "md5:6a2d740def5d13ea6096272546f6d41e",
    )
    df = pd.read_csv(path, header=None)
    df.columns = ["letter"] + [f"f{i}" for i in range(1, 17)]
    df["letter"] = df["letter"].astype("category")
    return df


def load_wine_quality_red() -> pd.DataFrame:
    """Wine Quality red (UCI): 12 numeric columns; recurrent in imputation benchmarks."""
    path = fetch_file(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "md5:2daeecee174368f8a33b82c8cccae3a5",
    )
    return pd.read_csv(path, sep=";")


def load_abalone() -> pd.DataFrame:
    """Abalone (UCI): mixed categorical (sex) and numeric shell measurements."""
    path = fetch_file(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
        "md5:a769fd0119787cac09158fe08971e480",
    )
    df = pd.read_csv(path, header=None)
    df.columns = ["sex", "length", "diameter", "height", "whole", "shucked", "viscera", "shell", "rings"]
    df["sex"] = df["sex"].astype("category")
    return df


def load_ionosphere() -> pd.DataFrame:
    """Ionosphere (UCI): 34 numeric radar features + binary class; used in MissForest."""
    path = fetch_file(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
        "md5:85649e5fb5b15fb9dab726c400be61fe",
    )
    df = pd.read_csv(path, header=None)
    df.columns = [f"f{i}" for i in range(34)] + ["class"]
    df["class"] = df["class"].astype("category")
    return df


def load_california_housing_df() -> pd.DataFrame:
    return fetch_california_housing(as_frame=True).frame.drop(columns=["MedHouseVal"])


def load_pems_bay_subset(n_rows: int = 8064, n_cols: int = 60) -> pd.DataFrame:
    """PEMS-BAY traffic speeds (Zenodo): standard traffic imputation benchmark (GRIN, SAITS).

    Subset (first 4 weeks of 5-minute data, first sensors) to keep runtime manageable.
    """
    from datafiller.datasets import load_pems_bay

    df = load_pems_bay()
    return df.iloc[:n_rows, :n_cols].astype(float)


def load_metr_la_subset(n_rows: int = 8064, n_cols: int = 60) -> pd.DataFrame:
    """METR-LA traffic speeds (Zenodo): standard traffic imputation benchmark (GRIN, BRITS).

    Zero readings mark sensor failures and are treated as missing. Subset as PEMS-BAY.
    """
    path = fetch_file(
        "https://zenodo.org/records/5724362/files/METR-LA.csv",
        "md5:7c463fcbd28ecc6c9cd93e1ca120aaaa",
    )
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df = df.rename_axis(index="time", columns="sensor_id").asfreq("5min")
    df = df.mask(df == 0.0)
    return df.iloc[:n_rows, :n_cols].astype(float)


def load_beijing_pm25() -> pd.DataFrame:
    """Beijing PM2.5 (UCI PRSA): hourly air quality, classic time series imputation dataset."""
    path = fetch_file(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv",
        "md5:83d51a811a081df89e691643af0edb29",
    )
    df = pd.read_csv(path)
    index = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.drop(columns=["No", "year", "month", "day", "hour"])
    df = df.set_axis(index, axis=0).rename_axis(index="time")
    df["cbwd"] = df["cbwd"].astype("category")
    return df.asfreq("h")


def load_etth1() -> pd.DataFrame:
    """ETTh1 (ETDataset): hourly electricity transformer temperature, used by SAITS et al."""
    path = fetch_file(
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "md5:8381763947c85f4be6ac456c508460d6",
    )
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").rename_axis(index="time")
    return df.asfreq("h")


TABULAR_DATASETS = [
    ("Diabetes", "numeric-only", lambda: load_diabetes(as_frame=True).frame.drop(columns=["target"])),
    ("Wine", "numeric-only", lambda: load_wine(as_frame=True).frame.drop(columns=["target"])),
    ("BreastCancer", "numeric-only", lambda: load_breast_cancer(as_frame=True).frame.drop(columns=["target"])),
    ("CaliforniaHousing", "numeric-only", load_california_housing_df),
    ("WineQualityRed", "numeric-only", load_wine_quality_red),
    ("Spambase", "numeric-only", load_spambase),
    ("LetterRecognition", "mixed", load_letter_recognition),
    ("Abalone", "mixed", load_abalone),
    ("Ionosphere", "mixed", load_ionosphere),
    ("Titanic", "mixed", lambda: __import__("datafiller.datasets", fromlist=["load_titanic"]).load_titanic()),
    ("SyntheticMixed", "mixed", lambda: load_synthetic_mixed_df(seed=SEED)),
]

# (name, kind, loader, call kwargs for TimeSeriesImputer.__call__)
TIMESERIES_DATASETS = [
    ("PemsBay", "numeric-only", load_pems_bay_subset, {"n_nearest_features": 100}),
    ("MetrLa", "numeric-only", load_metr_la_subset, {"n_nearest_features": 100}),
    ("BeijingPM25", "mixed", load_beijing_pm25, {}),
    ("ETTh1", "numeric-only", load_etth1, {}),
]

PATTERNS = [
    ("MAR_0.10", lambda shape, rng: make_mar_mask(shape, 0.10, rng)),
    ("Blocks_0.20x0.30", lambda shape, rng: make_block_mask(shape, 0.30, 0.20, rng)),
]


def drop_existing_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=0).reset_index(drop=True)


def make_mar_mask(shape: tuple[int, int], missing_ratio: float, rng: np.random.Generator) -> np.ndarray:
    return rng.random(shape) < missing_ratio


def make_block_mask(
    shape: tuple[int, int],
    frac_columns: float,
    block_length_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_rows, n_cols = shape
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    n_cols_to_mask = max(1, int(n_cols * frac_columns))
    cols = rng.choice(np.arange(n_cols), size=n_cols_to_mask, replace=False)
    block_length = max(1, int(n_rows * block_length_ratio))
    for col in cols:
        start = rng.integers(0, max(1, n_rows - block_length + 1))
        mask[start : start + block_length, col] = True
    return mask


def apply_mask(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    masked = df.copy()
    mask_df = pd.DataFrame(mask, columns=df.columns, index=df.index)
    for col in df.columns:
        if is_bool_dtype(masked[col].dtype):
            if masked[col].dtype != "boolean":
                masked[col] = masked[col].astype("boolean")
            masked.loc[mask_df[col], col] = pd.NA
        else:
            masked.loc[mask_df[col], col] = np.nan
    return masked


def numeric_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "mape": np.nan,
            "smape": np.nan,
            "median_ae": np.nan,
            "bias": np.nan,
            "nrmse_range": np.nan,
            "nrmse_std": np.nan,
        }

    errors = y_pred - y_true
    mse = np.mean(errors**2)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    median_ae = float(median_absolute_error(y_true, y_pred))
    bias = float(np.mean(errors))

    non_zero = y_true != 0
    mape = float(np.mean(np.abs(errors[non_zero] / y_true[non_zero])) * 100) if np.any(non_zero) else np.nan

    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(np.mean(2 * np.abs(errors) / denom) * 100) if np.any(denom != 0) else np.nan

    value_range = np.max(y_true) - np.min(y_true)
    nrmse_range = float(rmse / value_range) if value_range != 0 else np.nan
    std = np.std(y_true)
    nrmse_std = float(rmse / std) if std != 0 else np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "smape": smape,
        "median_ae": median_ae,
        "bias": bias,
        "nrmse_range": nrmse_range,
        "nrmse_std": nrmse_std,
    }


def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    valid = ~(y_true.isna() | y_pred.isna())
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) == 0:
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "precision_macro": np.nan,
            "recall_macro": np.nan,
            "f1_macro": np.nan,
            "mcc": np.nan,
            "kappa": np.nan,
        }

    all_classes = pd.Index(pd.concat([y_true, y_pred], axis=0).unique())
    y_true_codes = pd.Categorical(y_true, categories=all_classes).codes
    y_pred_codes = pd.Categorical(y_pred, categories=all_classes).codes

    labels = np.arange(len(all_classes))
    precision_macro = float(
        precision_score(y_true_codes, y_pred_codes, average="macro", labels=labels, zero_division=0)
    )
    recall_macro = float(recall_score(y_true_codes, y_pred_codes, average="macro", labels=labels, zero_division=0))
    f1_macro = float(f1_score(y_true_codes, y_pred_codes, average="macro", labels=labels, zero_division=0))
    balanced_accuracy = recall_macro

    return {
        "accuracy": float(accuracy_score(y_true_codes, y_pred_codes)),
        "balanced_accuracy": balanced_accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "mcc": float(matthews_corrcoef(y_true_codes, y_pred_codes)),
        "kappa": float(cohen_kappa_score(y_true_codes, y_pred_codes)),
    }


def is_categorical(series: pd.Series) -> bool:
    return bool(
        is_object_dtype(series.dtype)
        or is_string_dtype(series.dtype)
        or is_bool_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def evaluate(df_true: pd.DataFrame, df_pred: pd.DataFrame, mask: np.ndarray) -> dict:
    mask_df = pd.DataFrame(mask, columns=df_true.columns)

    numeric_true = []
    numeric_pred = []
    numeric_total = 0
    numeric_valid = 0

    cat_true = []
    cat_pred = []
    cat_total = 0
    cat_valid = 0

    for col in df_true.columns:
        col_mask = mask_df[col].to_numpy()
        if not np.any(col_mask):
            continue

        if is_categorical(df_true[col]):
            y_true = df_true.loc[col_mask, col].reset_index(drop=True)
            y_pred = df_pred.loc[col_mask, col].reset_index(drop=True)
            cat_true.append(y_true)
            cat_pred.append(y_pred)
            cat_total += len(y_true)
            cat_valid += int((~(y_true.isna() | y_pred.isna())).sum())
        else:
            y_true = df_true.loc[col_mask, col].to_numpy(dtype=float)
            y_pred = df_pred.loc[col_mask, col].to_numpy(dtype=float)
            numeric_true.append(y_true)
            numeric_pred.append(y_pred)
            numeric_total += len(y_true)
            numeric_valid += int(np.sum(np.isfinite(y_true) & np.isfinite(y_pred)))

    numeric_true_arr = np.concatenate(numeric_true) if numeric_true else np.array([])
    numeric_pred_arr = np.concatenate(numeric_pred) if numeric_pred else np.array([])
    cat_true_series = pd.concat(cat_true, axis=0, ignore_index=True) if cat_true else pd.Series(dtype=object)
    cat_pred_series = pd.concat(cat_pred, axis=0, ignore_index=True) if cat_pred else pd.Series(dtype=object)

    numeric_results = numeric_metrics(numeric_true_arr, numeric_pred_arr)
    categorical_results = classification_metrics(cat_true_series, cat_pred_series)

    coverage_numeric = float(numeric_valid / numeric_total) if numeric_total else np.nan
    coverage_categorical = float(cat_valid / cat_total) if cat_total else np.nan

    return {
        **numeric_results,
        **categorical_results,
        "coverage_numeric": coverage_numeric,
        "coverage_categorical": coverage_categorical,
        "masked_numeric": numeric_total,
        "masked_categorical": cat_total,
    }


def build_row(
    dataset_name: str,
    family: str,
    imputer_name: str,
    dataset_kind: str,
    pattern_name: str,
    df: pd.DataFrame,
    mask: np.ndarray,
    metrics: dict,
    elapsed: float,
) -> dict:
    numeric_cols = [col for col in df.columns if not is_categorical(df[col])]
    cat_cols = [col for col in df.columns if is_categorical(df[col])]
    missing_ratio = float(mask.mean()) if mask.size else 0.0

    return {
        "dataset": dataset_name,
        "family": family,
        "imputer": imputer_name,
        "dataset_kind": dataset_kind,
        "pattern": pattern_name,
        "rows": len(df),
        "cols": df.shape[1],
        "numeric_cols": len(numeric_cols),
        "categorical_cols": len(cat_cols),
        "missing_ratio": missing_ratio,
        "masked_total": int(mask.sum()),
        "masked_numeric": metrics["masked_numeric"],
        "masked_categorical": metrics["masked_categorical"],
        "time_seconds": elapsed,
        "coverage_numeric": metrics["coverage_numeric"],
        "coverage_categorical": metrics["coverage_categorical"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "mape": metrics["mape"],
        "smape": metrics["smape"],
        "median_ae": metrics["median_ae"],
        "bias": metrics["bias"],
        "nrmse_range": metrics["nrmse_range"],
        "nrmse_std": metrics["nrmse_std"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "mcc": metrics["mcc"],
        "kappa": metrics["kappa"],
    }


def run_tabular_benchmarks(rng: np.random.Generator, seed: int) -> list[dict]:
    rows = []
    for dataset_name, dataset_kind, loader in TABULAR_DATASETS:
        try:
            df = loader()
        except Exception as exc:
            print(f"Skipping dataset '{dataset_name}' due to error: {exc}")
            continue

        df = drop_existing_missing(df)
        if df.empty:
            print(f"Skipping dataset '{dataset_name}' because it is empty after dropna().")
            continue

        for pattern_name, mask_fn in PATTERNS:
            mask = mask_fn(df.shape, rng)
            df_missing = apply_mask(df, mask)

            imputer = MultivariateImputer(rng=seed)
            start = time.time()
            df_imputed = imputer(df_missing)
            elapsed = time.time() - start

            metrics = evaluate(df, df_imputed, mask)
            rows.append(
                build_row(
                    dataset_name,
                    "tabular",
                    "MultivariateImputer",
                    dataset_kind,
                    pattern_name,
                    df,
                    mask,
                    metrics,
                    elapsed,
                )
            )
            print(f"[tabular] {dataset_name} / {pattern_name}: {elapsed:.1f}s")
    return rows


def run_timeseries_benchmarks(rng: np.random.Generator, seed: int) -> list[dict]:
    rows = []
    for dataset_name, dataset_kind, loader, call_kwargs in TIMESERIES_DATASETS:
        try:
            df = loader()
        except Exception as exc:
            print(f"Skipping dataset '{dataset_name}' due to error: {exc}")
            continue

        for pattern_name, mask_fn in PATTERNS:
            # Preserve the time grid (rows are never dropped): mask only cells that are
            # actually observed, so every masked cell has a ground truth to score against.
            mask = mask_fn(df.shape, rng) & df.notna().to_numpy()
            df_missing = apply_mask(df, mask)

            imputer = TimeSeriesImputer(lags=TS_LAGS, rng=seed)
            start = time.time()
            df_imputed = imputer(df_missing, **call_kwargs)
            elapsed = time.time() - start

            metrics = evaluate(df, df_imputed, mask)
            rows.append(
                build_row(
                    dataset_name,
                    "timeseries",
                    "TimeSeriesImputer",
                    dataset_kind,
                    pattern_name,
                    df,
                    mask,
                    metrics,
                    elapsed,
                )
            )
            print(f"[timeseries] {dataset_name} / {pattern_name}: {elapsed:.1f}s")
    return rows


def run_benchmark(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = run_tabular_benchmarks(rng, seed)
    rows += run_timeseries_benchmarks(rng, seed)

    results = pd.DataFrame(rows)

    base_dir = Path(__file__).resolve().parents[1]
    docs_path = base_dir / "docs" / "_static" / "multivariate_benchmark_results.csv"
    results.to_csv(docs_path, index=False)
    print(f"Saved benchmark results to {docs_path}")

    return results


if __name__ == "__main__":
    run_benchmark()
