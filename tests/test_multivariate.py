import numpy as np
import pandas as pd
import pytest
from datafiller.multivariate import MultivariateImputer


@pytest.fixture
def nan_array():
    return np.array([[1, 2, 3, np.nan], [4, np.nan, 6, 7], [7, 8, 9, 10], [np.nan, 12, 13, 14]])


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
