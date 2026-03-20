:notoc: true

Performance
###########

This page tracks the repeatable benchmarking and profiling workflow used to optimize ``MultivariateImputer``.
It focuses on compute cost rather than user-facing accuracy tables in :doc:`benchmarks`.

Benchmark Scope
***************

The optimization session compared implementations on these workloads:

- ``numeric_mar_medium``: 5000x25 float32 array with 8% missing-at-random values.
- ``numeric_block_medium``: 5000x25 float32 array with contiguous missing blocks on 25% of columns.
- ``numeric_feature_selection``: 3500x80 float32 array with 8% missingness and ``n_nearest_features=12``.
- ``numeric_wide_light_missing``: 2500x120 float32 array with 4% missingness.
- ``mixed_dataframe_mar``: 4000-row mixed dataframe with numeric, categorical, string, and boolean columns.

Those benchmark inputs and their profiling outputs were generated locally during the optimization pass. They were
intentionally kept out of version control and are not part of the repository contents.

Strategy Tracker
****************

This ledger records the local branches used during the optimization pass.

.. list-table::
   :header-rows: 1

   * - Trial
     - Branch
     - Hypothesis
     - Status
     - Notes
   * - baseline
     - ``perf/tooling-registry``
     - Establish timing and profile baselines before changing implementations.
     - complete
     - Numeric profiling showed most time inside ``_impute_col`` with repeated ``optimask`` calls, ``FastRidge.fit``, and a large ``threading.wait`` cost from parallel Numba kernels.
   * - 01
     - ``perf/strategy-01-fast-ridge``
     - Reduce ``FastRidge`` overhead by avoiding extra dtype copies and diagonal-index allocation.
     - rejected
     - Helped medium numeric cases slightly but regressed the wide case.
   * - 02
     - ``perf/strategy-02-skip-optimask-clean``
     - Skip ``optimask`` when the candidate training subset is already NaN-free.
     - rejected
     - Small gains on default cases, but not stable enough and still regressed the wide case.
   * - 03
     - ``perf/strategy-03-serial-numba-kernels``
     - Remove ``parallel=True`` from the small Numba kernels to cut scheduler overhead.
     - kept
     - First major win: roughly 18 to 48 percent faster depending on the benchmark.
   * - 04
     - ``perf/strategy-04-serial-numba-plus-fast-ridge``
     - Keep the serial-kernel change and add the ``FastRidge`` cleanup on top.
     - winner
     - Best overall branch. Dedicated trial run improved the tracked cases by roughly 20 to 53 percent versus baseline.
   * - 05
     - ``perf/strategy-05-serial-numba-plus-skip-optimask``
     - Add the clean-subset ``optimask`` fast path on top of strategy 03.
     - rejected
     - Did not beat strategy 03 once the parallel overhead was already removed.
   * - 06
     - ``perf/strategy-06-reduce-index-copies``
     - Reduce index-array churn in ``imputer.py`` by using ``uint32`` indices and a shared validity mask.
     - rejected
     - Close to strategy 04 on some cases, but slower on the wide benchmark and not a consistent improvement.
   * - final
     - ``perf/final-report``
     - Ship the best code path and collect the final synthetic report.
     - complete
     - This branch keeps strategy 04 and the written summary of the optimization pass.

Synthetic Report
****************

Baseline Findings
=================

The baseline profiling pass showed three clear costs:

- ``MultivariateImputer._impute_col`` dominated end-to-end runtime.
- ``optimask`` and ``FastRidge.fit`` were the hottest Python-visible kernels inside that loop.
- The default numeric path spent a large share of time in ``threading.wait``, which pointed to Numba parallel scheduling overhead rather than useful work.

Winning Strategy
================

The winning implementation is the combination from ``perf/strategy-04-serial-numba-plus-fast-ridge``:

- remove ``parallel=True`` from the small Numba helper kernels in ``datafiller/_optimask.py`` and ``datafiller/multivariate/_numba_utils.py``
- keep ``FastRidge`` numerically identical while removing avoidable dtype copies and diagonal-index allocation in ``datafiller/estimators/ridge.py``

The final rerun on this branch kept the same correctness metrics as baseline across the registry and delivered these wall-clock improvements:

- ``numeric_block_medium``: 33.5 percent faster
- ``numeric_mar_medium``: 35.1 percent faster
- ``numeric_feature_selection``: 44.8 percent faster
- ``numeric_wide_light_missing``: 46.8 percent faster
- ``mixed_dataframe_mar``: 17.5 percent faster

The dedicated strategy-04 trial was even stronger on the four-case comparison set, peaking at 53.3 percent on
``numeric_feature_selection`` and 46.1 percent on ``numeric_wide_light_missing``. The final rerun is the more
conservative figure because it was collected together with the full suite on the report branch.

What Did Not Hold Up
====================

Several ideas looked plausible but were not worth carrying forward:

- The ``optimask`` clean-subset shortcut was not harmful in isolation, but it stopped mattering once the larger Numba scheduling overhead was removed.
- The ``FastRidge`` cleanup alone was too small and could regress the wide case.
- Reducing ``uint32`` index copies in ``imputer.py`` was measurable in profiles, but not enough to beat the simpler strategy-04 branch in wall-clock terms.

Validation Notes
================

The benchmark registry scores coverage, RMSE, MAE, and categorical accuracy, and those metrics remained unchanged across
the accepted strategy. Targeted direct validation scripts for ``FastRidge`` and ``MultivariateImputer`` behavior also
passed on every trial branch.

Some pytest runs were not reliable in this environment:

- mixed-type multivariate tests rely on remote dataset access
- broader pytest invocations timed out after the heavy Numba/imputer paths were loaded

Because of that, the primary acceptance criterion for the strategy branches is the repeatable benchmark and profiling
tooling added in this optimization pass.
