from datafiller import TimeSeriesImputer
from datafiller.datasets import add_contiguous_missing, add_mar, load_pems_bay


def main():
    df = load_pems_bay()
    df_to_ipmute = add_contiguous_missing(df=df, frac_columns=0.5, length=0.2)
    df_to_ipmute = add_mar(df=df_to_ipmute, nan_ratio=0.1)

    tsi = TimeSeriesImputer(
        lags=(-24, -6, -3 - 2 - 1, 1, 2, 3, 6, 24),
        rng=0,
        verbose=True,
        interpolate_gaps_less_than=1,
    )
    to_impute = df.sample(axis=1, n=15).columns.tolist()
    df_imputed = tsi(df=df_to_ipmute, n_nearest_features=50, cols_to_impute=to_impute)


if __name__ == "__main__":
    main()
