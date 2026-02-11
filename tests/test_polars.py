import numpy as np
import pytest

from datafiller.multivariate import MultivariateImputer


@pytest.fixture
def nan_array():
    rng = np.random.default_rng(0)
    n_samples = 300
    n_features = 6
    mean = np.linspace(0.0, 1.0, n_features)
    cov = np.fromfunction(lambda i, j: 0.5 ** np.abs(i - j), (n_features, n_features))
    x = rng.multivariate_normal(mean, cov, size=n_samples)
    n_nans = int(x.size * 0.10)
    nan_indices = rng.choice(x.size, size=n_nans, replace=False)
    x.flat[nan_indices] = np.nan
    return x


def test_multivariate_imputer_polars_dataframe_support(nan_array):
    pl = pytest.importorskip("polars")

    df = pl.DataFrame(nan_array, schema=[f"col_{i}" for i in range(nan_array.shape[1])])
    imputer = MultivariateImputer()
    imputed_df = imputer(df)

    assert isinstance(imputed_df, pl.DataFrame)
    assert imputed_df.columns == df.columns
    assert np.isnan(imputed_df.to_numpy()).sum() < np.isnan(df.to_numpy()).sum()


def test_multivariate_imputer_polars_preserves_dtypes():
    pl = pytest.importorskip("polars")

    df = pl.DataFrame(
        {
            "count": pl.Series([1, 2, None, 4, 5], dtype=pl.Int64),
            "value": pl.Series([10.0, 20.0, 30.0, None, 50.0], dtype=pl.Float64),
        }
    )
    imputer = MultivariateImputer(rng=0)
    imputed_df = imputer(df)

    assert imputed_df.dtypes == df.dtypes
