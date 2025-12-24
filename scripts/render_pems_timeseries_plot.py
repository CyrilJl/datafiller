import numpy as np

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

    df_out = df_imputed[[target_col]].rename(columns={target_col: "imputed"})
    df_out.insert(0, "ground_truth", ground_truth)
    df_out.to_csv("docs/_static/pems_bay_timeseries_imputation.csv", index_label="time")


if __name__ == "__main__":
    main()
