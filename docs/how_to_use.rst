:notoc: true

How to Use
##########

This guide provides detailed examples on how to use the ``MultivariateImputer`` and ``TimeSeriesImputer``. DataFiller targets a pragmatic middle ground for imputation: it does not aim to match the absolute performance of large deep learning models on complex masking patterns, but it is simple to fit, easy to adapt, and flexible to integrate into existing workflows. It is also significantly faster than scikit-learn's ``IterativeImputer``, which makes it well-suited for fast iteration and production use.

Multivariate Imputer
********************

The ``MultivariateImputer`` is the core of the library, designed to impute missing values in a 2D NumPy array or pandas DataFrame.
It automatically handles mixed numerical, boolean, and categorical/string columns by one-hot encoding non-numerical features internally so they can help impute other columns, then returning the original schema.

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

Titanic Mixed-Feature Example
=============================

``MultivariateImputer`` handles categorical, string, and boolean columns by one-hot encoding them internally and imputing missing labels with a classifier. The Titanic dataset provides a compact mixed-type example.

This example shows how categorical columns (such as ``sex`` or ``embarked``) are used as predictors for other features while their own missing values are imputed with a classifier.

.. code-block:: python

    from datafiller.datasets import load_titanic
    from datafiller import MultivariateImputer, ExtremeLearningMachine

    df = load_titanic()
    df.head(15)

.. include:: _static/titanic_head.md
   :parser: myst_parser.sphinx_

.. code-block:: python

    imputer = MultivariateImputer(regressor=ExtremeLearningMachine())
    df_imputed = imputer(df)
    df_imputed.head(15)

.. include:: _static/titanic_imputed_head.md
   :parser: myst_parser.sphinx_

Parameters
----------

The ``MultivariateImputer`` has a small set of knobs that control imputation behavior. For clarity, they are split into initialization
parameters (passed to ``MultivariateImputer(...)``) and call parameters (passed to ``imputer(...)``).

Initialization parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

``regressor``
    Numeric model for continuous targets. Defaults to a lightweight custom Ridge regressor.

``classifier``
    Model for categorical/string/boolean targets. Defaults to ``sklearn.linear_model.LogisticRegression``.

``verbose``
    Toggle progress output. Default is ``False``.

``min_samples_train``
    Minimum training size per column. Default is ``None`` (train whenever at least one sample is available).

``rng``
    Random generator or seed for reproducible feature sampling.

``scoring``
    Feature selection scoring. Use ``'default'`` or pass a callable ``(X, cols_to_impute) -> scores`` that returns a score matrix of
    shape ``(n_cols_to_impute, n_features)``.

Call parameters
~~~~~~~~~~~~~~~

``rows_to_impute``
    Target specific rows. Default is all rows.

``cols_to_impute``
    Target specific columns by index or name. Default is all columns.

``n_nearest_features``
    How many features are used for each imputation. Accepts an ``int`` count, a ``float`` fraction, or ``None`` for all features.

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
        regressor=RandomForestRegressor(n_estimators=10, random_state=0),
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

PEMS-BAY Example
================

This example loads the PEMS-BAY dataset, punches a large contiguous hole in one sensor's time series, adds 5% missing-at-random values to other sensors, and imputes the missing values using autoregressive lags and leads.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from datafiller import TimeSeriesImputer
    from datafiller.datasets import add_mar, load_pems_bay

    df = load_pems_bay()
    rng = np.random.default_rng(0)
    target_col = rng.choice(df.columns)
    ground_truth = df[target_col].copy()

    df_missing = df.copy()
    n_rows = len(df_missing)
    hole_length = int(n_rows * 0.2)
    start = n_rows // 2 - hole_length // 2
    end = start + hole_length
    df_missing.loc[df_missing.index[start:end], target_col] = np.nan

    other_cols = df_missing.columns.drop(target_col)
    np.random.seed(0)
    df_missing.loc[:, other_cols] = add_mar(df_missing[other_cols], nan_ratio=0.05)

    ts_imputer = TimeSeriesImputer(lags=[1, 2, 3, -1, -2, -3], rng=0)
    df_imputed = ts_imputer(df_missing, cols_to_impute=[target_col], n_nearest_features=75)

.. raw:: html

    <div id="pems-bay-timeseries-plot" style="width: 100%; height: 420px;"></div>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <script>
    (function () {
      const csvUrl = "https://raw.githubusercontent.com/CyrilJl/datafiller/main/docs/_static/pems_bay_timeseries_imputation.csv";
      fetch(csvUrl)
        .then((response) => response.text())
        .then((text) => {
          const lines = text.trim().split("\n");
          const header = lines.shift().split(",");
          const idxTime = header.indexOf("time");
          const idxTruth = header.indexOf("ground_truth");
          const idxImputed = header.indexOf("imputed");
          const time = [];
          const ground = [];
          const imputed = [];

          for (const line of lines) {
            const cols = line.split(",");
            time.push(cols[idxTime]);
            ground.push(parseFloat(cols[idxTruth]));
            imputed.push(parseFloat(cols[idxImputed]));
          }

          const data = [
            {
              x: time,
              y: ground,
              name: "ground truth",
              mode: "lines",
              line: { color: "#2a9d8f", width: 2.4 },
            },
            {
              x: time,
              y: imputed,
              name: "imputed",
              mode: "lines",
              line: { color: "#e76f51", width: 1.8 },
            },
          ];
          const maxGround = Math.max(...ground);
          const maxImputed = Math.max(...imputed);
          const maxValue = Math.max(maxGround, maxImputed);
          const yMax = maxValue > 0 ? maxValue * 1.05 : 1;
          const layout = {
            margin: { t: 20, r: 20, b: 40, l: 50 },
            xaxis: { title: "time" },
            yaxis: { title: "value", range: [0, yMax] },
            legend: { orientation: "h" },
          };
          Plotly.newPlot("pems-bay-timeseries-plot", data, layout, {
            responsive: true,
          });
        });
    })();
    </script>

Parameters
----------

Initialization parameters include ``lags`` for autoregressive features (positive integers create lags like `t-1`, negative integers create leads like `t+1`, default `(1,)`), ``regressor`` for the numeric model (default ``FastRidge()``), ``min_samples_train`` to require a minimum training size (default ``None``), ``rng`` for reproducibility, ``verbose`` for logging, ``scoring`` for feature selection, and ``interpolate_gaps_less_than`` to linearly interpolate short gaps before model-based imputation (default ``None``).

Call parameters (``__call__``) include ``rows_to_impute`` and ``cols_to_impute`` to target subsets (both default to all), ``n_nearest_features`` to limit features used for imputation, and ``before``/``after`` to restrict the time window by timestamp (ignored when ``rows_to_impute`` is provided).
