How to Use
##########

This guide provides detailed examples on how to use the ``MultivariateImputer`` and ``TimeSeriesImputer``.

Multivariate Imputer
********************

The ``MultivariateImputer`` is the core of the library, designed to impute missing values in a 2D NumPy array or pandas DataFrame.

Basic Example
=============

Here is a simple example of how to use the ``MultivariateImputer``.

.. code-block:: python

    import numpy as np
    from datafiller import MultivariateImputer

    # Create a matrix with some missing values
    X = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, np.nan, 7.0, 8.0],
        [9.0, 10.0, 11.0, np.nan],
        [13.0, 14.0, 15.0, 16.0],
    ])

    # Initialize the imputer
    imputer = MultivariateImputer()

    # Impute the missing values
    X_imputed = imputer(X)

    print(X_imputed)

Parameters
----------

The ``MultivariateImputer`` has several parameters that can be tuned to control the imputation process.

**Initialization Parameters**

*   **estimator**: The regressor model to use for imputation. It should be a lightweight model, as it is fitted many times. By default, a custom Ridge implementation is used.
*   **verbose**: Controls the verbosity of the imputer. If ``True``, it will print progress bars. Defaults to ``False``.
*   **min_samples_train**: The minimum number of samples required to train a model for a given column. If, after the imputation, some values are still missing, it is likely that no training set with at least `min_samples_train` samples could be found. Defaults to ``None``, which means that a model will be trained if at least one sample is available.
*   **rng**: A seed for the random number generator, which is used for reproducible feature sampling. Defaults to ``None``.
*   **scoring**: The scoring function to use for feature selection. If 'default', the default scoring function is used. If a callable, it must take two arguments as input: the data matrix `X` (np.ndarray of shape `(n_samples, n_features)`) and the columns to impute `cols_to_impute` (np.ndarray of shape `(n_cols_to_impute,)`), and return a score matrix of shape `(n_cols_to_impute, n_features)`. Defaults to `'default'`.

**Call Parameters**

*   **rows_to_impute**: The specific rows to impute. Can be a list of indices. If ``None``, all rows are considered.
*   **cols_to_impute**: The specific columns to impute. Can be a list of indices or column names (for DataFrames). If ``None``, all columns are considered.
*   **n_nearest_features**: The number of nearest features to use for imputation. If it's an ``int``, it's the absolute number of features. If it's a ``float``, it's the fraction of total features. If ``None``, all features are used.

Advanced Usage
--------------

Here is a more advanced example that shows how to use some of the parameters.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from datafiller.multivariate import MultivariateImputer
    from sklearn.ensemble import RandomForestRegressor

    # Create a DataFrame with missing values
    data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, 4, 5],
        'C': [1, 2, 3, np.nan, 5],
        'D': [1, 2, 3, 4, np.nan]
    }
    df = pd.DataFrame(data)

    # Initialize the imputer with a RandomForestRegressor
    imputer = MultivariateImputer(
        estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        verbose=1,
        rng=0
    )

    # Impute only column 'A' and 'B', using only 2 nearest features
    df_imputed = imputer(
        df,
        cols_to_impute=['A', 'B'],
        n_nearest_features=2
    )

    print(df_imputed)

Custom Scoring Function
~~~~~~~~~~~~~~~~~~~~~~~

You can provide a custom scoring function to control how the imputer selects features for imputation. The scoring function should
take the data matrix `X` and the columns to impute `cols_to_impute` as input, and return a score matrix.

Here is an example of a custom scoring function that simply returns a random score matrix.

.. code-block:: python

    import numpy as np
    from datafiller.multivariate import MultivariateImputer

    def random_scoring(X, cols_to_impute):
        n_cols_to_impute = len(cols_to_impute)
        n_features = X.shape[1]
        return np.random.rand(n_cols_to_impute, n_features)

    # Create a matrix with missing values
    X = np.array([
        [1.0, 2.0, np.nan, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, np.nan, 11.0, 12.0],
    ])

    # Initialize the imputer with the custom scoring function
    imputer = MultivariateImputer(
        scoring=random_scoring,
        rng=42
    )

    # Impute using 2 nearest features, selected based on the random scores
    X_imputed = imputer(X, n_nearest_features=2)

    print(X_imputed)

Time Series Imputer
********************

The ``TimeSeriesImputer`` is a wrapper around the ``MultivariateImputer`` that is specifically designed for time series data.

Basic Example
=============

The ``TimeSeriesImputer`` requires a pandas DataFrame with a ``DatetimeIndex`` that has a defined frequency.

.. code-block:: python

    import pandas as pd
    import numpy as np
    from datafiller import TimeSeriesImputer

    # Create a time series DataFrame with missing values
    rng = pd.date_range('2023-01-01', periods=20, freq='D')
    data = {
        'feature1': np.sin(np.arange(20) * 0.5),
        'feature2': np.cos(np.arange(20) * 0.5),
    }
    df = pd.DataFrame(data, index=rng)

    # Add some missing values
    df.loc['2023-01-05', 'feature1'] = np.nan
    df.loc['2023-01-10', 'feature2'] = np.nan
    df.loc['2023-01-15', 'feature1'] = np.nan

    # Initialize the imputer with lags [1, 2] and leads [-1, -2]
    ts_imputer = TimeSeriesImputer(lags=[1, 2, -1, -2])
    df_imputed = ts_imputer(df)

    print(df_imputed)

Parameters
----------

**Initialization Parameters**

*   **lags**: An iterable of integers specifying the lags and leads to create as autoregressive features. Positive integers create lags (e.g., `t-1`), and negative integers create leads (e.g., `t+1`). Defaults to `(1,)`.
*   **estimator**: The regressor model to use for imputation. Defaults to ``FastRidge()``.
*   **min_samples_train**: The minimum number of samples required to train a model. Defaults to ``None``, which means that a model will be trained if at least one sample is available.
*   **rng**: A seed for the random number generator. Defaults to ``None``.
*   **verbose**: Controls the verbosity. Defaults to ``0``.
*   **scoring**: The scoring function for feature selection. Defaults to `'default'`.
*   **interpolate_gaps_less_than**: The maximum length of gaps to interpolate linearly before model-based imputation. If ``None``, no linear interpolation is performed. Defaults to `None`.

**Call Parameters (``__call__``)**

*   **rows_to_impute**: The indices of rows to impute. If ``None``, all rows are considered.
*   **cols_to_impute**: The indices or names of columns to impute. If ``None``, all columns are considered.
*   **n_nearest_features**: The number of features to use for imputation.
*   **before**: A timestamp-like object. If specified, only rows before this timestamp are imputed.
*   **after**: A timestamp-like object. If specified, only rows after this timestamp are imputed.

Advanced Usage
--------------

This example shows how to use the ``TimeSeriesImputer`` to impute missing values in a specific time window.

.. code-block:: python

    import pandas as pd
    import numpy as np
    from datafiller.timeseries import TimeSeriesImputer

    # Create a time series DataFrame with missing values
    rng = pd.date_range('2023-01-01', periods=20, freq='D')
    data = {
        'feature1': np.sin(np.arange(20) * 0.5),
        'feature2': np.cos(np.arange(20) * 0.5),
    }
    df = pd.DataFrame(data, index=rng)

    # Add some missing values
    df.loc['2023-01-05', 'feature1'] = np.nan
    df.loc['2023-01-10', 'feature2'] = np.nan
    df.loc['2023-01-15', 'feature1'] = np.nan

    # Initialize the imputer with lags and linear interpolation
    ts_imputer = TimeSeriesImputer(
        lags=[1, 2, -1, -2],
        interpolate_gaps_less_than=3
    )

    # Impute only the missing values that occured before 2023-01-12
    df_imputed = ts_imputer(
        df,
        before='2023-01-12'
    )

    print(df_imputed)
