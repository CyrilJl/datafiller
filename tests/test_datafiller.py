import numpy as np
from datafiller import DataFiller

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
    data[start_row:start_row + n_missing_rows, start_col:start_col + n_missing_cols] = np.nan
    return data

def test_datafiller():
    n_samples = 100
    n_features = 10
    mean = np.zeros(n_features)
    cov = np.eye(n_features)
    data = generate_data(n_samples, n_features, mean, cov)
    data_missing_random = generate_missing_at_random(data.copy(), 0.1)
    data_missing_consecutive = generate_consecutive_missing(data.copy(), 10, 3)

    data_filler = DataFiller()
    data_imputed_random = data_filler(data_missing_random)
    data_imputed_consecutive = data_filler(data_missing_consecutive)

    assert np.isnan(data_imputed_random).sum() == 0
    assert np.isnan(data_imputed_consecutive).sum() == 0
