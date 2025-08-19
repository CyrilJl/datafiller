import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from datafiller import MultivariateImputer
from datafiller.multivariate._scoring import preimpute
from datafiller.multivariate._utils import (
    _dataframe_cols_to_impute_to_indices,
    _dataframe_rows_to_impute_to_indices,
    _process_to_impute,
)


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


def test_process_to_impute():
    assert np.array_equal(_process_to_impute(5, None), np.arange(5))
    assert np.array_equal(_process_to_impute(5, 2), np.array([2]))
    assert np.array_equal(_process_to_impute(5, [1, 3]), np.array([1, 3]))


def test_preimpute():
    x = np.array([[1, np.nan], [3, 5]], dtype=float)
    xp = preimpute(x)
    assert np.array_equal(xp, [[1, 5], [3, 5]])


def test_validate_input_errors():
    imputer = MultivariateImputer()
    x = np.random.rand(10, 5)

    class NotNumpy:
        pass

    with pytest.raises(ValueError, match="x must be a numpy array"):
        imputer(NotNumpy())

    with pytest.raises(ValueError, match="x must be a 2D array"):
        imputer(np.random.rand(10))

    with pytest.raises(ValueError, match="x must have a numeric dtype"):
        imputer(np.array([["a", "b"], ["c", "d"]]))

    with pytest.raises(ValueError, match="x cannot contain infinity"):
        x_inf = x.copy()
        x_inf[0, 0] = np.inf
        imputer(x_inf)

    with pytest.raises(ValueError, match="rows_to_impute must have an integer dtype"):
        imputer(x, rows_to_impute=np.array([0.5, 1.5]))

    with pytest.raises(ValueError, match="rows_to_impute must be a list of integers between 0 and 9"):
        imputer(x, rows_to_impute=np.array([10]))

    with pytest.raises(ValueError, match="rows_to_impute must be a list of integers between 0 and 9"):
        imputer(x, rows_to_impute=[10])

    with pytest.raises(ValueError, match="rows_to_impute must be a list of integers between 0 and 9"):
        imputer(x, rows_to_impute=[0, "a"])

    with pytest.raises(ValueError, match="cols_to_impute must be a list of integers between 0 and 4"):
        imputer(x, cols_to_impute=[5])

    with pytest.raises(ValueError, match="cols_to_impute must be a list of integers between 0 and 4"):
        imputer(x, cols_to_impute=[0, "a"])

    with pytest.raises(ValueError, match="If n_nearest_features is a float, it must be in"):
        imputer(x, n_nearest_features=1.1)

    with pytest.raises(ValueError, match="If n_nearest_features is a float, it must be in"):
        imputer(x, n_nearest_features=0.0)

    with pytest.raises(ValueError, match="n_nearest_features resulted in 0 features"):
        imputer(np.random.rand(10, 10), n_nearest_features=0.01)

    with pytest.raises(ValueError, match="n_nearest_features must be an int or float"):
        imputer(x, n_nearest_features="not a number")

    with pytest.raises(ValueError, match="n_nearest_features must be between 1 and 5"):
        imputer(x, n_nearest_features=6)

    with pytest.raises(ValueError, match="n_nearest_features must be between 1 and 5"):
        imputer(x, n_nearest_features=0)


def test_get_sampled_cols_zero_scores(mocker):
    mocker.patch(
        "numpy.random.RandomState",
        return_value=mocker.Mock(choice=lambda a, size, replace, p: a[:size]),
    )
    imputer = MultivariateImputer()
    scores = np.zeros((1, 5))
    # with p=None, it should do a uniform choice
    sampled_cols = imputer._get_sampled_cols(5, 0, 3, scores, 0)
    assert len(sampled_cols) == 3
    assert len(np.unique(sampled_cols)) == 3
    assert np.array_equal(sampled_cols, np.array([1, 2, 3]))  # due to mock


def test_imputer_no_nans_in_col_to_impute():
    imputer = MultivariateImputer()
    x = np.array([[1, np.nan], [2, 4]], dtype=float)
    x_imputed = imputer(x, cols_to_impute=0)
    assert np.array_equal(x, x_imputed, equal_nan=True)


def test_imputer_all_nans_in_col_to_impute():
    imputer = MultivariateImputer()
    x = np.array([[np.nan, 1], [np.nan, 2]], dtype=float)
    x_imputed = imputer(x, cols_to_impute=0)
    assert np.array_equal(x, x_imputed, equal_nan=True)


def test_dataframe_to_indices_single_item():
    df_num_index = pd.DataFrame(np.random.rand(3, 3), index=[10, 20, 30], columns=[1, 2, 3])
    rows = _dataframe_rows_to_impute_to_indices(20, df_num_index.index)
    assert np.array_equal(rows, [1])
    cols = _dataframe_cols_to_impute_to_indices(3, df_num_index.columns)
    assert np.array_equal(cols, [2])


def test_dataframe_to_indices_tuple():
    df = pd.DataFrame(np.random.rand(3, 3), index=["r1", "r2", "r3"], columns=["c1", "c2", "c3"])
    rows = _dataframe_rows_to_impute_to_indices(("r1", "r3"), df.index)
    assert np.array_equal(rows, [0, 2])
    cols = _dataframe_cols_to_impute_to_indices(("c1", "c3"), df.columns)
    assert np.array_equal(cols, [0, 2])


def test_reproducible_imputation():
    data = np.array(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]],
        dtype=float,
    )
    data[1, 1] = np.nan
    data[2, 3] = np.nan
    data[4, 0] = np.nan

    imputer1 = MultivariateImputer(estimator=Ridge(random_state=0), min_samples_train=2, rng=42)
    imputer2 = MultivariateImputer(estimator=Ridge(random_state=0), min_samples_train=2, rng=42)
    imputer3 = MultivariateImputer(estimator=Ridge(random_state=0), min_samples_train=2, rng=43)

    imputed1 = imputer1(data.copy(), n_nearest_features=3)
    imputed2 = imputer2(data.copy(), n_nearest_features=3)
    imputed3 = imputer3(data.copy(), n_nearest_features=3)

    assert np.array_equal(imputed1, imputed2)
    assert not np.array_equal(imputed1, imputed3)


def test_multivariate_imputer_min_samples_train_none():
    data = np.array([[1, 2], [np.nan, 4]], dtype=float)

    # With min_samples_train=None (default), it should impute using the single available row.
    imputer_none = MultivariateImputer()
    imputed_none = imputer_none(data.copy())
    assert not np.isnan(imputed_none).any()

    # With min_samples_train=2, it should not be able to impute.
    imputer_2 = MultivariateImputer(min_samples_train=2)
    imputed_2 = imputer_2(data.copy())
    assert np.isnan(imputed_2).any()
