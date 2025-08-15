import numpy as np
import pandas as pd
import pytest
from datafiller import MultivariateImputer


def generate_data(n_samples, n_features, mean, cov):
    return np.random.multivariate_normal(mean, cov, n_samples)


def generate_missing_at_random(data, missing_rate):
    n_samples, n_features = data.shape
    mask = np.random.rand(n_samples, n_features) < missing_rate

    # Ensure that no column is entirely missing
    all_missing_cols = np.all(mask, axis=0)
    for col_idx in np.where(all_missing_cols)[0]:
        mask[0, col_idx] = False

    data[mask] = np.nan
    return data


def generate_consecutive_missing(data, n_missing_rows, n_missing_cols):
    n_samples, n_features = data.shape
    start_row = np.random.randint(0, n_samples - n_missing_rows)
    start_col = np.random.randint(0, n_features - n_missing_cols)
    data[start_row : start_row + n_missing_rows, start_col : start_col + n_missing_cols] = np.nan
    return data


def test_multivariate_imputer():
    n_samples = 100
    n_features = 10
    mean = np.zeros(n_features)
    cov = np.eye(n_features)
    data = generate_data(n_samples, n_features, mean, cov)
    data_missing_random = generate_missing_at_random(data.copy(), 0.1)
    data_missing_consecutive = generate_consecutive_missing(data.copy(), 10, 3)

    data_filler = MultivariateImputer(min_samples_train=10)
    data_imputed_random = data_filler(data_missing_random)
    data_imputed_consecutive = data_filler(data_missing_consecutive)

    assert np.isnan(data_imputed_random).sum() == 0
    assert np.isnan(data_imputed_consecutive).sum() == 0


def test_multivariate_imputer_with_dataframe():
    n_samples = 100
    n_features = 5
    data = np.random.rand(n_samples, n_features)
    data[5:10, 2] = np.nan  # some nans

    df = pd.DataFrame(
        data, columns=[f"col_{i}" for i in range(n_features)], index=[f"row_{i}" for i in range(n_samples)]
    )

    imputer = MultivariateImputer(min_samples_train=10)
    df_imputed = imputer(df)

    assert isinstance(df_imputed, pd.DataFrame)
    assert df_imputed.shape == df.shape
    assert (df_imputed.columns == df.columns).all()
    assert (df_imputed.index == df.index).all()
    assert not df_imputed.isnull().values.any()


def test_multivariate_imputer_dataframe_with_labels():
    data = np.array([[1, 2, 3, 1], [4, np.nan, 6, 4], [7, 8, np.nan, 7], [1, 2, 3, np.nan]], dtype=float)

    df = pd.DataFrame(data, columns=["A", "B", "C", "D"], index=["r1", "r2", "r3", "r4"])

    # original nan positions: (1,1), (2,2), (3,3)

    imputer = MultivariateImputer(min_samples_train=2)

    # Case 1: impute column 'B' only
    df_imputed_col = imputer(df.copy(), cols_to_impute="B")
    assert not np.isnan(df_imputed_col.loc["r2", "B"])
    assert np.isnan(df_imputed_col.loc["r3", "C"])  # should still be nan
    assert np.isnan(df_imputed_col.loc["r4", "D"])  # should still be nan

    # Case 2: impute row 'r3' only
    df_imputed_row = imputer(df.copy(), rows_to_impute="r3")
    assert np.isnan(df_imputed_row.loc["r2", "B"])  # should still be nan
    assert not np.isnan(df_imputed_row.loc["r3", "C"])
    assert np.isnan(df_imputed_row.loc["r4", "D"])  # should still be nan

    # Case 3: impute col 'C' for row 'r3'
    df_imputed_cell = imputer(df.copy(), rows_to_impute="r3", cols_to_impute="C")
    assert np.isnan(df_imputed_cell.loc["r2", "B"])  # should still be nan
    assert not np.isnan(df_imputed_cell.loc["r3", "C"])
    assert np.isnan(df_imputed_cell.loc["r4", "D"])  # should still be nan


def test_multivariate_imputer_dataframe_label_not_found():
    data = np.array([[1, 2], [np.nan, 4]], dtype=float)
    df = pd.DataFrame(data, columns=["A", "B"], index=["r1", "r2"])

    imputer = MultivariateImputer(min_samples_train=2)

    with pytest.raises(ValueError, match="Column labels not found"):
        imputer(df, cols_to_impute=["C"])

    with pytest.raises(ValueError, match="Row labels not found"):
        imputer(df, rows_to_impute=["r3"])
