:notoc: true

Benchmarks
##########

This page summarizes benchmark results for the ``MultivariateImputer`` and the ``TimeSeriesImputer`` across multiple open datasets and missingness patterns.

Benchmark Table
***************

The table below is rendered with DataTables, the same third-party display library used in :doc:`how_to_use`.

.. raw:: html

    <link href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css" rel="stylesheet">
    <style>
      .benchmark-table-wrap {
        margin-top: 8px;
      }
      .benchmark-table-status {
        margin: 6px 0 12px;
        font-weight: 600;
      }
    </style>
    <div class="benchmark-table-wrap">
      <table id="multivariate-benchmark-table" class="display" style="width: 100%"></table>
    </div>
    <script src="https://unpkg.com/papaparse@5.4.1/papaparse.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
    <script>
    (function () {
      const tableId = "multivariate-benchmark-table";
      const el = document.getElementById(tableId);
      if (!el) return;
      el.insertAdjacentHTML("afterend", "<div class='benchmark-table-status'>Loading benchmark table...</div>");
      const statusEl = el.nextElementSibling;

      const baseUrl = "https://raw.githubusercontent.com/CyrilJl/datafiller/main/docs/_static/";
      const csvUrl = baseUrl + "multivariate_benchmark_results.csv";

      fetch(csvUrl)
        .then((response) => response.text())
        .then((text) => {
          const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
          const fields = parsed.meta.fields || [];
          const columns = fields.map((field) => ({
            title: field,
            data: field,
          }));
          $(el).DataTable({
            data: parsed.data,
            columns: columns,
            pageLength: 10,
            lengthMenu: [10, 25, 50, 100],
            order: [],
            scrollX: true,
          });
          if (statusEl) statusEl.remove();
        })
        .catch(() => {
          if (statusEl) statusEl.textContent = "Failed to load benchmark results.";
        });
    })();
    </script>

Methodology
***********

Benchmarks are computed by injecting synthetic missingness into each dataset, imputing, and scoring only the masked entries against the ground truth. Two missingness patterns are evaluated:

- MAR_0.10: 10% missing-at-random across all cells.
- Blocks_0.20x0.30: contiguous blocks covering 20% of the rows in 30% of the columns.

Two dataset families are covered (see the ``family`` and ``imputer`` columns):

- **Tabular** datasets are imputed with ``MultivariateImputer``. Rows with pre-existing missing values are dropped first so every masked cell has a ground truth. Datasets include numeric-only sets widely used in the imputation literature (Diabetes, Wine, Breast Cancer, California Housing, Wine Quality red, Spambase) and mixed numeric/categorical sets (Letter Recognition, Abalone, Ionosphere, Titanic, plus a synthetic mixed dataset).
- **Time series** datasets are imputed with ``TimeSeriesImputer`` (lags/leads 1-3, default time features). The time grid is preserved: rows are never dropped, and synthetic missingness is only injected into observed cells, so pre-existing gaps stay untouched and every masked cell can be scored. Datasets are standard time series imputation benchmarks: the PEMS-BAY and METR-LA traffic-speed datasets (first four weeks of 5-minute data, first 60 sensors, imputed with ``n_nearest_features=100``; zero readings in METR-LA are treated as missing), the Beijing PM2.5 hourly air-quality dataset (mixed: numeric measurements plus the categorical wind direction), and the ETTh1 electricity-transformer dataset.

All external datasets are downloaded and cached with ``pooch`` from their canonical open sources (UCI Machine Learning Repository, Zenodo, the ETDataset repository) and verified against pinned checksums.

Metrics are split by data type: regression metrics (RMSE, MAE, R2, MAPE, SMAPE, median AE, bias, normalized RMSE) for numeric columns and classification metrics (accuracy, balanced accuracy, macro precision/recall/F1, MCC, Cohen's kappa) for categorical columns. Coverage reports the fraction of masked values that received finite predictions.
