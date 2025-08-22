<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/CyrilJl/datafiller/main/docs/_static/datafiller_light.svg">
  <img alt="datafiller logo" src="https://raw.githubusercontent.com/CyrilJl/datafiller/main/docs/_static/datafiller_dark.svg" width="50%" height="50%">
</picture>

[![PyPI version](https://badge.fury.io/py/datafiller.svg)](https://badge.fury.io/py/datafiller)
[![CI Pipeline](https://github.com/CyrilJl/datafiller/actions/workflows/ci-pipeline.yml/badge.svg)](https://github.com/CyrilJl/datafiller/actions/workflows/ci-pipeline.yml)
[![codecov](https://codecov.io/github/CyrilJl/datafiller/graph/badge.svg?token=PXK2523PL9)](https://codecov.io/github/CyrilJl/datafiller)
[![Documentation Status](https://readthedocs.org/projects/datafiller/badge/?version=latest)](https://datafiller.readthedocs.io/en/latest/?badge=latest)


</div>

**DataFiller** is a Python library for imputing missing values in datasets. It provides a flexible and powerful way to handle missing data in both numerical arrays and time series data.

## Key Features

- **Model-Based Imputation**: Uses machine learning models (like linear regression) to predict and fill missing values.
- **Time Series Support**: A dedicated ``TimeSeriesImputer`` that automatically creates lagged and lead features for imputation.
- **Efficient**: Leverages Numba for performance-critical sections.
- **Smart Feature Selection**: Finds the optimal subset of data to use for training imputation models.
- **Scikit-Learn Compatible**: Integrates with the scikit-learn ecosystem.

## Installation

You can install DataFiller using pip:

```bash
pip install datafiller
```

## Basic Usage

### Imputing a NumPy Array

The ``MultivariateImputer`` can be used to fill missing values (`NaN`) in a 2D NumPy array.

```python
import numpy as np
from datafiller import MultivariateImputer

# Create a matrix with missing values
X = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, np.nan, 7.0, 8.0],
    [9.0, 10.0, 11.0, np.nan],
    [13.0, 14.0, 15.0, 16.0],
])

# Initialize the imputer and fill the missing values
imputer = MultivariateImputer()
X_imputed = imputer(X)

print("Original Matrix:")
print(X)
print("\nImputed Matrix:")
print(X_imputed)
```

### Imputing a Time Series DataFrame

The ``TimeSeriesImputer`` is designed to work with pandas DataFrames that have a ``DatetimeIndex``. It automatically creates autoregressive features (lags and leads) to improve imputation accuracy.

```python
import pandas as pd
import numpy as np
from datafiller import TimeSeriesImputer

# Create a time series DataFrame with missing values
rng = pd.date_range('2023-01-01', periods=10, freq='D')
data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [10, 9, np.nan, 7, 6, 5, np.nan, 3, 2, 1],
}
df = pd.DataFrame(data, index=rng)

# Initialize the imputer with lags and leads
# Use t-1 and t+1 to impute missing values
ts_imputer = TimeSeriesImputer(lags=[1, -1])
df_imputed = ts_imputer(df)

print("Original DataFrame:")
print(df)
print("\nImputed DataFrame:")
print(df_imputed)
```

## How It Works

DataFiller uses a model-based imputation strategy. For each column containing missing values, it trains a regression model using the other columns as features. The rows used for training are carefully selected to be the largest, most complete rectangular subset of the data, which is found using the ``optimask`` algorithm. This ensures that the training data is of the highest possible quality, leading to more accurate imputations.

For more details, see the [documentation](https://datafiller.readthedocs.io/).
