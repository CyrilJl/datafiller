<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/CyrilJl/datafiller/main/docs/_static/datafiller_light.svg">
  <img alt="datafiller logo" src="https://raw.githubusercontent.com/CyrilJl/datafiller/main/docs/_static/datafiller_dark.svg" width="50%" height="50%">
</picture>

[![PyPI version](https://badge.fury.io/py/datafiller.svg)](https://badge.fury.io/py/datafiller)
[![Conda version](https://anaconda.org/conda-forge/datafiller/badges/version.svg)](https://anaconda.org/conda-forge/datafiller)
[![Documentation Status](https://readthedocs.org/projects/datafiller/badge/?version=latest)](https://datafiller.readthedocs.io/en/latest/?badge=latest)


</div>

# DataFiller

**DataFiller** is a Python library for imputing missing values in datasets. It provides a flexible and powerful way to handle missing data in both numerical arrays and time series data.

## Why DataFiller

DataFiller is a pragmatic imputation tool: it is unlikely to match the absolute performance of large deep learning approaches on complex masking patterns, but it is much simpler to fit, easier to adapt, and more flexible to plug into existing workflows. It is also significantly faster than scikit-learn's ``IterativeImputer``, which makes it a good choice when you need strong results with tight iteration cycles.

## Key Features

Key features include model-based imputation with lightweight models, mixed data support with one-hot encoding and label recovery, a dedicated ``TimeSeriesImputer`` with lag/lead and calendar features for missing-at-random values and contiguous timestamp gaps, performance-critical sections accelerated by Numba, smart feature selection for training subsets, and scikit-learn compatibility.

## Installation

Install DataFiller using pip or conda:

```bash
pip install datafiller
```

Install the optional Polars integration with:

```bash
pip install "datafiller[polars]"
```

```bash
conda install -c conda-forge datafiller
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

The ``TimeSeriesImputer`` is designed to work with pandas DataFrames that have a ``DatetimeIndex``. It automatically creates autoregressive features (lags and leads) to improve imputation accuracy, and can infer a regular frequency to reinsert missing timestamp blocks before imputation.

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

### Imputing a Mixed DataFrame with Categorical Features

Categorical columns are one-hot encoded and used as predictors for other columns, while missing categorical values are imputed with a classifier and mapped back to labels.

```python
from datafiller.datasets import load_titanic
from datafiller import MultivariateImputer, ExtremeLearningMachine

df = load_titanic()
imputer = MultivariateImputer(regressor=ExtremeLearningMachine())
df_imputed = imputer(df)
```

### Using Polars

Both imputers accept eager Polars DataFrames and return Polars DataFrames. The multivariate imputer uses integer row positions and
column names for targeted imputation:

```python
import polars as pl
from datafiller import MultivariateImputer

df = pl.DataFrame({"temperature": [18.0, None, 21.0], "weather": ["sun", "rain", "sun"]})
df_imputed = MultivariateImputer()(df)
```

For time series, configure the Date or Datetime column explicitly. The timestamp column is retained in the result and missing timestamps
inside a regular series are reinserted before imputation:

```python
from datetime import datetime, timedelta

import polars as pl
from datafiller import TimeSeriesImputer

df = pl.DataFrame({
    "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(24)],
    "value": [float(i) if i != 12 else None for i in range(24)],
})
imputer = TimeSeriesImputer(time_column="timestamp", lags=[1, -1])
df_imputed = imputer(df)
```

Polars `LazyFrame` input is not supported because model fitting requires materializing the data; call `collect()` first.

### GPU Acceleration (optional)

The default ridge models can be solved as batched GPU operations by passing `device="cuda"`. This pays off when many columns are imputed on medium-to-large matrices (7-12x measured on an RTX 4060 when imputing all 250 columns of 30k-100k row matrices); imputed values match the CPU path up to float32 rounding.

GPU support relies on PyTorch, which is an optional dependency:

```bash
pip install datafiller[gpu]
# or install a build matching your CUDA setup: https://pytorch.org/get-started/locally/
```

```python
imputer = MultivariateImputer(device="cuda")
X_imputed = imputer(X)
```

Everything else is unchanged: with the default `device=None`, PyTorch is never imported and the pure NumPy/Numba CPU path runs. Categorical targets, custom regressors, and patterns with too few complete rows transparently fall back to the CPU implementation.

## How It Works

DataFiller uses a model-based imputation strategy. For each column containing missing values, it trains a model using the other columns as features. Categorical, boolean, and string columns are one-hot encoded for feature construction, so they can drive the imputation of numerical targets, and are imputed with a classifier before being mapped back to the original labels. Rows to impute are grouped by their pattern of observed features and get one model per pattern; with the default ridge regressor these models are solved efficiently from a shared Gram matrix.

For each pattern, the training data is selected along a three-step path:

1. **Complete rows** — the rows fully observed on the pattern's features are used directly when at least `min_samples_train` of them exist (default 20, calibrated on real and synthetic datasets).
2. **optimask** — otherwise, the [optimask](https://github.com/CyrilJl/OptiMask) algorithm finds the largest complete rectangular subset of the data, trading some feature columns for more training rows, and prefers rectangles that keep at least `min_samples_train` rows.
3. **Fallback** — the rare cells that still cannot get a model are filled with the column mean (most frequent category for categoricals) by default; pass `fallback=None` to leave them as NaN instead.

This ensures each model is trained on the highest-quality data available while still guaranteeing a fully imputed output.

For more details, see the [documentation](https://datafiller.readthedocs.io/).
