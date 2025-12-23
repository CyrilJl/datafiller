from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datafiller import TimeSeriesImputer
from datafiller.datasets import add_mar, load_pems_bay


def main() -> None:
    df = load_pems_bay()

    rng = np.random.default_rng(0)
    target_col = rng.choice(df.columns)
    ground_truth = df[target_col].copy()

    df_missing = df.copy()
    n_rows = len(df_missing)
    hole_length = max(1, int(n_rows * 0.2))
    hole_center = n_rows // 3
    start = max(0, hole_center - hole_length // 2)
    end = start + hole_length
    df_missing.loc[df_missing.index[start:end], target_col] = np.nan

    other_cols = df_missing.columns.drop(target_col)
    np.random.seed(0)
    df_missing.loc[:, other_cols] = add_mar(df_missing[other_cols], nan_ratio=0.05)

    ts_imputer = TimeSeriesImputer(lags=[1, 2, 3, -1, -2, -3], rng=0)
    df_imputed = ts_imputer(df_missing, cols_to_impute=[target_col], n_nearest_features=75)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ground_truth.index, ground_truth.values, label="Ground truth", linewidth=1.2)
    ax.plot(df_imputed.index, df_imputed[target_col].values, label="Imputed", linewidth=0.8)
    margin = max(1, hole_length // 5)
    left = df_missing.index[max(0, start - margin)]
    right = df_missing.index[min(n_rows - 1, end + margin)]
    ax.set_xlim(left, right)
    ax.set_title(f"PEMS-BAY imputation for sensor {target_col}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Speed")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("docs/_static/pems_bay_timeseries_imputation.png", dpi=150)
    df_out = df_imputed[[target_col]].rename(columns={target_col: "imputed"})
    df_out.insert(0, "ground_truth", ground_truth)
    df_out.to_csv("docs/_static/pems_bay_timeseries_imputation.csv", index_label="time")


if __name__ == "__main__":
    main()
