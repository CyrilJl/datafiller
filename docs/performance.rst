:notoc: true

Performance
###########

This page tracks the repeatable benchmarking and profiling workflow used to optimize ``MultivariateImputer``.
It focuses on compute cost rather than user-facing accuracy tables in :doc:`benchmarks`.

Benchmark Registry
******************

The registry lives in ``scripts/_multivariate_perf_registry.py`` and defines the cases used for optimization work:

- ``numeric_mar_medium``: 5000x25 float32 array with 8% missing-at-random values.
- ``numeric_block_medium``: 5000x25 float32 array with contiguous missing blocks on 25% of columns.
- ``numeric_feature_selection``: 3500x80 float32 array with 8% missingness and ``n_nearest_features=12``.
- ``numeric_wide_light_missing``: 2500x120 float32 array with 4% missingness.
- ``mixed_dataframe_mar``: 4000-row mixed dataframe with numeric, categorical, string, and boolean columns.

The suite intentionally uses synthetic and local-only cases so it can run without remote dataset downloads and stay stable
across optimization branches.

Tooling
*******

Run the benchmark suite:

.. code-block:: bash

   python scripts/run_multivariate_benchmarks.py --repeat 3 --warmup 1 --output docs/_static/performance/baseline.csv

Run a focused profile for one or more registry entries:

.. code-block:: bash

   python scripts/profile_multivariate.py --benchmark numeric_mar_medium --benchmark mixed_dataframe_mar

The benchmark runner records wall-clock time plus lightweight correctness metrics. The profiling runner writes text and
JSON summaries under ``docs/_static/performance/profiles/`` using ``cProfile`` and ``tracemalloc``.

Strategy Tracker
****************

This ledger is meant to be updated as optimization branches are evaluated.

.. list-table::
   :header-rows: 1

   * - Trial
     - Branch
     - Hypothesis
     - Status
     - Notes
   * - baseline
     - current branch
     - Establish timing and profile baselines before changing implementations.
     - complete
     - Numeric profiling shows the cost concentrated in repeated ``optimask`` calls and per-pattern ridge fits.
   * - tbd
     - pending
     - Fill in after branch evaluation.
     - pending
     - Use the benchmark registry outputs as the source of truth.
