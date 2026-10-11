:notoc: true

How to Use
##########

This guide provides detailed examples on how to use the ``MultivariateImputer`` and ``TimeSeriesImputer``. DataFiller targets a pragmatic middle ground for imputation: it does not aim to match the absolute performance of large deep learning models on complex masking patterns, but it is simple to fit, easy to adapt, and flexible to integrate into existing workflows. It is also significantly faster than scikit-learn's ``IterativeImputer``, which makes it well-suited for fast iteration and production use.

Multivariate Imputer
********************

The ``MultivariateImputer`` is the core of the library, designed to impute missing values in a 2D NumPy array or eager pandas or Polars DataFrame.
It automatically handles mixed numerical, boolean, and categorical/string columns by one-hot encoding non-numerical features internally so they can help impute other columns, then returning the original schema.

Polars DataFrames
=================

Install the optional integration with ``pip install "datafiller[polars]"``. Polars inputs return Polars outputs, preserve column order and
supported dtypes, and treat both ``null`` and floating-point ``NaN`` as missing. ``rows_to_impute`` contains integer row positions and
``cols_to_impute`` contains column names. Numeric, Boolean, String, Categorical, and Enum columns are supported. LazyFrames must be
collected before imputation.

.. code-block:: python

    import polars as pl
    from datafiller import MultivariateImputer

    df = pl.DataFrame({
        "value": [1.0, None, 3.0, 4.0],
        "group": ["a", "a", None, "b"],
    })
    df_imputed = MultivariateImputer()(df)

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
        targetEl.scrollLeft(sourceEl.scrollLeft());
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
``min_samples_train``, ``fallback``, and ``verbose`` to control feature selection, reproducibility, training thresholds, last-resort
filling, and logging. Call parameters include ``rows_to_impute`` and ``cols_to_impute`` to target subsets and ``n_nearest_features`` to
limit the features used per imputation; setting ``n_nearest_features`` is recommended to reduce computation time. For a complete list and
full descriptions, see the :doc:`api` reference.

Training-data selection follows a three-step path for each missingness pattern — **complete rows → optimask → fallback**: rows fully
observed on the pattern's features are used when at least ``min_samples_train`` (default 20) of them exist; otherwise `optimask` finds the
largest NaN-free rectangle, preferring ones that keep at least ``min_samples_train`` rows; and the rare cells that still cannot get a model
are filled with the column mean (most frequent category for categoricals) with the default ``fallback="simple"`` — pass ``fallback=None`` to leave them as NaN.
See :doc:`algorithm` for details.

Using scikit-learn pipelines
============================

Both imputers provide ``fit`` and ``transform`` and can be placed in a scikit-learn pipeline. ``fit`` is a no-op: every
``transform`` call fits models on the data it receives. The imputers do not learn models on a training dataset and reuse them
to fill a separate test dataset. Account for this behaviour when designing cross-validation or forecasting evaluations.

Observed values are preserved in the output. With the default fallback, selected missing cells are filled whenever their
column has an observed value; entirely missing columns remain missing.

GPU Acceleration
================

Both imputers accept an optional ``device`` parameter (e.g. ``device="cuda"``) that solves the default ridge models as batched GPU
operations instead of a per-pattern Python loop. This is most beneficial when many columns are imputed on medium-to-large matrices,
where it can be an order of magnitude faster; imputed values match the CPU path up to float32 rounding.

GPU support relies on PyTorch, which is an optional dependency and is only imported when a device is requested:

.. code-block:: bash

    pip install datafiller[gpu]

.. code-block:: python

    from datafiller import MultivariateImputer

    imputer = MultivariateImputer(device="cuda")
    X_imputed = imputer(X)

The default PyPI build of PyTorch ships CUDA support on Linux; on Windows, install a CUDA-enabled build by following
`pytorch.org/get-started <https://pytorch.org/get-started/locally/>`_. With the default ``device=None`` the pure NumPy/Numba CPU path
runs and PyTorch is never imported. Categorical targets, custom regressors, and patterns with fewer than ``min_samples_train``
complete rows transparently fall back to the CPU implementation.

Time Series Imputer
********************

Row positions refer to the regularized output grid. Pandas and Polars accept timestamp values as an alternative to positions.
Pandas column selectors may use names or non-negative positions in the original columns; Polars selectors use names.
Generated lag and calendar features cannot be targeted. An explicit ``rows_to_impute`` takes precedence over ``before`` and
``after``. These selectors also restrict linear interpolation: surrounding observations provide the interpolation context,
but only selected cells are filled.

The ``TimeSeriesImputer`` is a wrapper around the ``MultivariateImputer`` that is specifically designed for time series data. It can infer a regular ``DatetimeIndex`` frequency, reinsert missing timestamp rows inside the observed range, and add low-cost calendar features that remain observed through contiguous gaps.

With Polars, pass an eager DataFrame and configure its Date or Datetime column through ``time_column``. The timestamp column remains in
the returned Polars DataFrame. Row selection accepts positions on the regularized output grid or timestamp values; column selection uses
names and cannot target the timestamp column.

.. code-block:: python

    import polars as pl
    from datafiller import TimeSeriesImputer

    df = pl.DataFrame({
        "timestamp": pl.datetime_range(
            start=pl.datetime(2024, 1, 1),
            end=pl.datetime(2024, 1, 2),
            interval="1h",
            eager=True,
        ),
        "value": [float(i) for i in range(25)],
    }).with_columns(
        pl.when(pl.int_range(pl.len()) == 12).then(None).otherwise("value").alias("value")
    )

    df_imputed = TimeSeriesImputer(time_column="timestamp", lags=[1, -1])(df)

Lags per column
===============

An iterable ``lags`` gives every column the same lags, so with ``n`` columns and ``k`` lags each model sees ``n * (1 + k)`` series
features. Often only some columns benefit from lags, typically the target's own history. A mapping lags only the columns it names; the
others get none. Its keys are column labels, not positions: ``{0: (1,)}`` refers to a column named ``0`` (a name that is not in the
data raises an error). With Polars, the ``time_column`` cannot be a key.

.. code-block:: python

    TimeSeriesImputer(lags=(1, 2))                                        # every column
    TimeSeriesImputer(lags={"y": (1, 2, 3, 24)})                          # only y
    TimeSeriesImputer(lags={"y": ("1h", "1D"), "x": (-1,)})               # durations, and a lead on x
    TimeSeriesImputer(lags={**{c: (1,) for c in df}, "y": range(1, 25)})  # a default plus an override

A lag is an integer number of steps or a duration (``"24h"``, ``pd.Timedelta(minutes=30)``). Durations are converted with the nominal length
of the inferred index frequency (``"1D"`` is one step of a daily index, even across a daylight saving change), must be a non-zero
whole number of steps, and need a unit (``"24"`` is rejected); negative durations are leads. Generated features are named after the
number of steps, ``f"{column}_lag_{steps}"``: on an hourly index ``"1D"`` gives ``y_lag_24`` and ``"-1h"`` gives ``y_lag_-1``. A column
given the same lag twice (``(1, "1h")`` on an hourly index) raises an error.

After a call, ``feature_names_`` lists the feature matrix the models were fitted on: the original columns, then the lagged columns, then
the calendar features. An iterable produces one block of every column per lag (``x_lag_1, y_lag_1, x_lag_2, y_lag_2``); a mapping
follows the column order of the data, then the order of each column's lags.

The mapping form turns the imputer into a direct multi-horizon forecaster. Blank the horizon after a cut-off and impute it with ``after=``:
at ``h`` steps after the cut-off, lag ``k`` of the target is observed only if ``h <= k``, so each horizon step gets its own missingness
pattern and its own model. Lagging only the target keeps those models small; lagging every exogenous column multiplies the features and
can overfit short histories.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from datafiller import TimeSeriesImputer

    # Four weeks of hourly measurements `y` and two exogenous model forecasts.
    rng = np.random.default_rng(0)
    index = pd.date_range("2025-01-01", periods=24 * 28, freq="h")
    daily = np.sin(2 * np.pi * np.arange(len(index)) / 24)
    df = pd.DataFrame(
        {
            "y": daily + 0.3 * rng.normal(size=len(index)).cumsum() / 10,
            "forecast_a": daily + 0.2 * rng.normal(size=len(index)),
            "forecast_b": daily + 0.4 * rng.normal(size=len(index)),
        },
        index=index,
    )

    # Blank the 15 hours after the cut-off: they are the horizon to forecast.
    cutoff = index[-16]
    df.loc[df.index > cutoff, "y"] = np.nan

    # Lag the target only; give the forecasts a one-hour lead.
    imputer = TimeSeriesImputer(lags={"y": (1, 2, 3, "1D"), "forecast_a": (-1,), "forecast_b": (-1,)})
    imputed = imputer(df, cols_to_impute=["y"], after=cutoff)
    forecast = imputed.loc[imputed.index > cutoff, "y"]  # the 15 forecast hours
    print(imputer.feature_names_)
    # ['y', 'forecast_a', 'forecast_b', 'y_lag_1', 'y_lag_2', 'y_lag_3', 'y_lag_24',
    #  'forecast_a_lag_-1', 'forecast_b_lag_-1', '__time_trend', ...]

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
    df_missing.loc[:, other_cols] = add_mar(df_missing[other_cols], nan_ratio=0.05, rng=0)

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
leads like `t+1`; durations and per-column mappings are accepted, see `Lags per column`_), ``regressor`` for the numeric model, ``interpolate_gaps_less_than`` to pre-fill short gaps, ``add_time_features`` to
include deterministic calendar/trend predictors, and the shared controls ``scoring``, ``rng``, ``min_samples_train``, ``fallback``, and
``verbose`` (the training-data selection path described above applies unchanged). Call parameters include ``rows_to_impute`` and ``cols_to_impute`` to target
subsets, ``n_nearest_features`` to limit features used for imputation (recommended to reduce computation time), and ``before``/``after`` to
restrict the time window. For a complete list and full descriptions, see the :doc:`api` reference.
