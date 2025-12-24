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

.. code-block:: python

    imputer = MultivariateImputer(regressor=ExtremeLearningMachine())
    df_imputed = imputer(df)
    df_imputed.head(15)

.. raw:: html

    <link href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css" rel="stylesheet">
    <style>
      .titanic-table-wrap {
        display: flex;
        gap: 16px;
        align-items: flex-start;
        flex-wrap: wrap;
      }
      .titanic-table-block {
        flex: 1 1 420px;
        min-width: 320px;
      }
      .titanic-table-title {
        margin: 8px 0 6px;
        font-weight: 600;
      }
    </style>
    <div class="titanic-table-wrap">
      <div class="titanic-table-block">
        <div class="titanic-table-title">Original Titanic (titanic.csv)</div>
        <table id="titanic-table" class="display" style="width: 100%"></table>
      </div>
      <div class="titanic-table-block">
        <div class="titanic-table-title">Imputed Titanic (titanic_imputed.csv)</div>
        <table id="titanic-imputed-table" class="display" style="width: 100%"></table>
      </div>
    </div>
    <script src="https://unpkg.com/papaparse@5.4.1/papaparse.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
    <script>
    (function () {
      let tableLeft = null;
      let tableRight = null;
      let isSyncingPage = false;
      let isSyncingScroll = false;

      function buildTable(containerId, csvUrl) {
        const el = document.getElementById(containerId);
        if (!el) return;
        el.insertAdjacentHTML("afterend", "<div class='titanic-table-status'>Loading table...</div>");
        const statusEl = el.nextElementSibling;
        fetch(csvUrl)
          .then((response) => response.text())
          .then((text) => {
            const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
            const fields = parsed.meta.fields || [];
            const columns = fields.map((field) => ({
              title: field,
              data: field,
            }));
            const table = $(el).DataTable({
              data: parsed.data,
              columns: columns,
              pageLength: 10,
              lengthMenu: [10, 25, 50, 100],
              order: [],
              scrollX: true,
              scrollY: "420px",
              scrollCollapse: true,
            });
            if (statusEl) statusEl.remove();
            return table;
          })
          .then((table) => {
            if (!table) return;
            if (containerId === "titanic-table") {
              tableLeft = table;
            } else if (containerId === "titanic-imputed-table") {
              tableRight = table;
            }
            if (tableLeft && tableRight) {
              syncTables(tableLeft, tableRight);
            }
          })
          .catch(() => {
            if (statusEl) statusEl.textContent = "Failed to load table data.";
          });
      }

      function syncTables(left, right) {
        left.on("page.dt", () => syncPage(left, right));
        right.on("page.dt", () => syncPage(right, left));

        const leftBody = $(left.table().container()).find(".dataTables_scrollBody");
        const rightBody = $(right.table().container()).find(".dataTables_scrollBody");

        leftBody.on("scroll", () => syncScroll(leftBody, rightBody));
        rightBody.on("scroll", () => syncScroll(rightBody, leftBody));
      }

      function syncPage(source, target) {
        if (isSyncingPage) return;
        isSyncingPage = true;
        const pageInfo = source.page.info();
        target.page(pageInfo.page).draw(false);
        isSyncingPage = false;
      }

      function syncScroll(sourceEl, targetEl) {
        if (isSyncingScroll) return;
        isSyncingScroll = true;
        targetEl.scrollTop(sourceEl.scrollTop());
        isSyncingScroll = false;
      }

      const baseUrl = "https://raw.githubusercontent.com/CyrilJl/datafiller/main/docs/_static/";
      buildTable("titanic-table", baseUrl + "titanic.csv");
      buildTable("titanic-imputed-table", baseUrl + "titanic_imputed.csv");
    })();
    </script>

Parameters
----------

The main initialization parameters are ``regressor`` and ``classifier`` to set the numeric/categorical models, plus ``scoring``, ``rng``,
``min_samples_train``, and ``verbose`` to control feature selection, reproducibility, training thresholds, and logging. Call parameters
include ``rows_to_impute`` and ``cols_to_impute`` to target subsets and ``n_nearest_features`` to limit the features used per imputation.
For a complete list and full descriptions, see the :doc:`api` reference.

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
              line: { color: "#1f77b4", width: 2 },
            },
            {
              x: time,
              y: imputed,
              name: "imputed",
              mode: "lines",
              line: { color: "#ff7f0e", width: 1.4 },
            },
          ];
          const layout = {
            margin: { t: 20, r: 20, b: 40, l: 50 },
            xaxis: { title: "time" },
            yaxis: { title: "value", range: [0, 80] },
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

Initialization parameters include ``lags`` for autoregressive features (positive integers create lags like `t-1`, negative integers create
leads like `t+1`), ``regressor`` for the numeric model, ``interpolate_gaps_less_than`` to pre-fill short gaps, and the shared controls
``scoring``, ``rng``, ``min_samples_train``, and ``verbose``. Call parameters include ``rows_to_impute`` and ``cols_to_impute`` to target
subsets, ``n_nearest_features`` to limit features used for imputation, and ``before``/``after`` to restrict the time window. For a complete
list and full descriptions, see the :doc:`api` reference.
