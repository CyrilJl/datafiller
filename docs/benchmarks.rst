:notoc: true

Benchmarks
##########

This page summarizes benchmark results for the ``MultivariateImputer`` across multiple datasets and missingness patterns.

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

Benchmarks are computed by taking each dataset, dropping rows with existing missing values (to preserve ground truth), injecting synthetic missingness, imputing, and scoring only the masked entries. Two missingness patterns are evaluated:

- MAR_0.10: 10% missing-at-random across all cells.
- Blocks_0.20x0.30: contiguous blocks covering 20% of the rows in 30% of the columns.

Datasets include three numeric-only scikit-learn tabular datasets (Diabetes, Wine, Breast Cancer), plus mixed-type Titanic and a synthetic mixed dataset.
Metrics are split by data type: regression metrics (RMSE, MAE, R2, MAPE, SMAPE, median AE, bias, normalized RMSE) for numeric columns and classification metrics (accuracy, balanced accuracy, macro precision/recall/F1, MCC, Cohen's kappa) for categorical columns. Coverage reports the fraction of masked values that received finite predictions.
