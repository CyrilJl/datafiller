##########
How to Use
##########

This guide provides detailed examples on how to use the `MultivariateImputer` and `TimeSeriesImputer`.

***********************
Multivariate Imputer
***********************

The `MultivariateImputer` is the core of the library, designed to impute missing values in a 2D NumPy array.

Basic Example
=============

Here is a simple example of how to use the `MultivariateImputer`.

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


Using a Different Estimator
===========================

You can use any scikit-learn compatible regressor as the estimator for imputation. For example, you can use a `RandomForestRegressor`.

.. code-block:: python

    import numpy as np
    from datafiller import MultivariateImputer
    from sklearn.ensemble import RandomForestRegressor

    X = np.random.rand(100, 10)
    X[np.random.choice(X.size, 100, replace=False)] = np.nan

    # Initialize the imputer with a RandomForestRegressor
    imputer = MultivariateImputer(estimator=RandomForestRegressor(n_estimators=10))
    X_imputed = imputer(X)

    print("Imputation successful:", not np.isnan(X_imputed).any())


********************
Time Series Imputer
********************

The `TimeSeriesImputer` is a wrapper around the `MultivariateImputer` that is specifically designed for time series data.

Basic Example
=============

The `TimeSeriesImputer` requires a pandas DataFrame with a `DatetimeIndex` that has a defined frequency.

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
