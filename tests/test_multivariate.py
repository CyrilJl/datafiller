import numpy as np
import pandas as pd
import pytest

from datafiller.datasets import load_titanic
from datafiller.multivariate import MultivariateImputer


@pytest.fixture
def nan_array():
    rng = np.random.default_rng(0)
    n_samples = 9
    n_features = 10
    mean = np.linspace(0.0, 1.0, n_features)
    cov = np.fromfunction(lambda i, j: 0.5 ** np.abs(i - j), (n_features, n_features))
    x = rng.multivariate_normal(mean, cov, size=n_samples)
    n_nans = int(x.size * 0.10)
    nan_indices = rng.choice(x.size, size=n_nans, replace=False)
    x.flat[nan_indices] = np.nan
    return x


@pytest.fixture
def titanic_mixed_df():
    df = load_titanic()

    cols = ["sex", "age", "fare", "embarked", "deck"]
    df = df[cols].copy()
    df["sex"] = df["sex"].astype("category")
    df["embarked"] = df["embarked"].astype("category")
    df["deck"] = df["deck"].astype("category")
    return df


def test_multivariate_imputer_less_nans(nan_array):
    imputer = MultivariateImputer()
    imputed_array = imputer(nan_array)
    assert np.isnan(imputed_array).sum() < np.isnan(nan_array).sum()


def test_multivariate_imputer_dataframe_support(nan_array):
    df = pd.DataFrame(nan_array, columns=[f"col_{i}" for i in range(nan_array.shape[1])])
    imputer = MultivariateImputer()
    imputed_df = imputer(df)
    assert isinstance(imputed_df, pd.DataFrame)
    assert np.isnan(imputed_df.values).sum() < np.isnan(df.values).sum()


def test_multivariate_imputer_categorical_dataframe_support(titanic_mixed_df):
    imputer = MultivariateImputer(rng=0)
    imputed_df = imputer(titanic_mixed_df)
    assert list(imputed_df.columns) == list(titanic_mixed_df.columns)
    assert imputed_df["embarked"].isna().sum() < titanic_mixed_df["embarked"].isna().sum()
    assert imputed_df["deck"].isna().sum() < titanic_mixed_df["deck"].isna().sum()
    assert set(imputed_df["sex"].dropna().unique()).issubset({"male", "female"})
    assert set(imputed_df["embarked"].dropna().unique()).issubset({"C", "Q", "S"})
    assert set(imputed_df["deck"].dropna().unique()).issubset({"A", "B", "C", "D", "E", "F", "G", "T"})


def test_multivariate_imputer_cols_to_impute(nan_array):
    imputer = MultivariateImputer()
    imputed_array = imputer(nan_array, cols_to_impute=[1, 3])
    assert np.isnan(imputed_array[:, 0]).sum() == np.isnan(nan_array[:, 0]).sum()
    assert np.isnan(imputed_array[:, 1]).sum() == 0
    assert np.isnan(imputed_array[:, 2]).sum() == np.isnan(nan_array[:, 2]).sum()
    assert np.isnan(imputed_array[:, 3]).sum() == 0


def test_multivariate_imputer_rows_to_impute(nan_array):
    imputer = MultivariateImputer()
    imputed_array = imputer(nan_array, rows_to_impute=[1, 3])
    assert np.isnan(imputed_array[0, :]).sum() == np.isnan(nan_array[0, :]).sum()
    assert np.isnan(imputed_array[1, :]).sum() == 0
    assert np.isnan(imputed_array[2, :]).sum() == np.isnan(nan_array[2, :]).sum()
    assert np.isnan(imputed_array[3, :]).sum() == 0


def test_multivariate_imputer_min_samples_train(nan_array):
    imputer = MultivariateImputer(min_samples_train=10)
    imputed_array = imputer(nan_array)
    # With a high min_samples_train, no imputation should happen
    assert np.isnan(imputed_array).sum() == np.isnan(nan_array).sum()


def test_multivariate_imputer_boolean_support():
    df = pd.DataFrame(
        {
            "flag": pd.Series([True, False, None, True, None], dtype="boolean"),
            "value": [1.0, 2.0, 3.0, np.nan, 5.0],
        }
    )
    imputer = MultivariateImputer(rng=0)
    imputed_df = imputer(df)
    assert imputed_df["flag"].isna().sum() < df["flag"].isna().sum()
    assert imputed_df["flag"].dtype == "boolean"
    assert set(imputed_df.columns) == {"flag", "value"}


def test_multivariate_imputer_preserves_numeric_dtypes():
    df = pd.DataFrame(
        {
            "count": pd.Series([1, 2, None, 4, 5], dtype="Int64"),
            "value": pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], dtype="float64"),
        }
    )
    imputer = MultivariateImputer(rng=0)
    imputed_df = imputer(df)
    assert imputed_df["count"].dtype == df["count"].dtype
    assert imputed_df["value"].dtype == df["value"].dtype


@pytest.mark.parametrize("use_df", [False, True])
def test_multivariate_imputer_n_nearest_features_tracking(nan_array, use_df):
    if use_df:
        x = pd.DataFrame(nan_array, columns=[f"col_{i}" for i in range(nan_array.shape[1])])
        cols_with_nans = x.columns[x.isnull().any()].tolist()
    else:
        x = nan_array
        cols_with_nans = np.where(np.isnan(x).any(axis=0))[0]

    imputer = MultivariateImputer(rng=0)
    n_nearest_features = 2
    imputer(x, n_nearest_features=n_nearest_features)

    assert imputer.imputation_features_ is not None
    assert set(imputer.imputation_features_.keys()) == set(cols_with_nans)

    for col, features in imputer.imputation_features_.items():
        if use_df:
            assert isinstance(features, list)
            assert all(isinstance(f, str) for f in features)
        else:
            assert isinstance(features, np.ndarray)
        assert len(features) <= n_nearest_features
        assert col not in features
