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
