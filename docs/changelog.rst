Changelog
=========

Unreleased
----------

- ``TimeSeriesImputer`` now accepts categorical, string, object, and boolean columns in pandas input: categorical targets are imputed with the configured classifier, and their lagged/lead copies participate as features alongside numeric ones. Polars time series input remains numeric-only.
- The benchmark suite (``scripts/multivariate_benchmark.py`` and the docs benchmarks page) now covers open datasets commonly used to assess imputation algorithms: tabular UCI/sklearn datasets (Spambase, Letter Recognition, Wine Quality red, Abalone, Ionosphere, California Housing) imputed with ``MultivariateImputer``, and time series benchmarks (PEMS-BAY, METR-LA, Beijing PM2.5, ETTh1) imputed with ``TimeSeriesImputer``.

v0.3.2 (2026-07-19)
-------------------

- Optional Polars support: ``MultivariateImputer`` now accepts eager Polars DataFrames with numeric, Boolean, String, Categorical, and Enum columns, while ``TimeSeriesImputer`` accepts a Polars Date/Datetime ``time_column`` and preserves it when regularizing timestamp gaps. Both imputers return Polars output for Polars input; install with ``datafiller[polars]``.
- **Behavior change:** ``min_samples_train`` now defaults to 20 (was 1). A benchmark across real and synthetic datasets showed that models fitted on a handful of rows produce imputations worse than plain column means once missingness reaches ~25%; thresholds of 10-20 were consistently best. Pass ``min_samples_train=1`` to restore the previous behavior.
- New ``fallback`` parameter on ``MultivariateImputer`` and ``TimeSeriesImputer``: cells no model could impute (their missingness pattern never reached ``min_samples_train`` training rows) are now filled with the column mean (most frequent category for categoricals) by default (``fallback="simple"``) instead of being left NaN. Pass ``fallback=None`` to keep NaNs.
- ``optimask`` now takes the caller's row requirement into account (``min_rows``): among the NaN-free rectangles on its pareto front, it prefers ones with at least ``min_samples_train`` rows over strictly larger ones with fewer, and only falls back to the unconstrained maximum-area choice when the constraint is infeasible. This removes most of the imputation-coverage loss previously caused by strict ``min_samples_train`` values on heavily missing data, and is byte-identical to the previous selection whenever the constraint does not bind.
- Optional GPU acceleration: ``MultivariateImputer`` and ``TimeSeriesImputer`` accept a ``device`` parameter (e.g. ``"cuda"``) that solves all missingness patterns of a column as batched tensor operations. Requires the new ``datafiller[gpu]`` extra (PyTorch), which stays entirely optional: with the default ``device=None`` PyTorch is never imported. Imputed values match the CPU path up to float32 rounding; categorical targets, custom regressors, and patterns with too few complete rows keep using the CPU implementation.

v0.3.1 (2026-07-04)
-------------------

- ``ExtremeLearningMachine`` rewritten for speed and memory: BLAS-backed projection instead of a numba kernel, and chunked Gram-based fits/predicts above 65,536 rows so the full hidden matrix is never materialized (about 2.6x faster standalone fit, 7x lower peak memory on large fits).
- Fixed a correctness bug where refitting the same ``ExtremeLearningMachine`` instance on inputs of varying widths (as ``MultivariateImputer`` does per missingness pattern) read past or truncated the random projection matrix, producing non-finite or corrupted imputations. Projections are now sampled lazily per input width and cached, seeded reproducibly.
- Fixed an accuracy issue that made ``ExtremeLearningMachine`` systematically underperform ``FastRidge`` in typical imputation workloads: projection weights are now fan-in scaled, and a new ``min_samples_per_feature`` parameter (default 5) caps the hidden width by the number of training samples so small per-pattern fits are no longer severely underdetermined.

v0.3 (2026-07-04)
-----------------

- TimeSeriesImputer can infer a regular DatetimeIndex frequency and reinsert missing timestamp rows inside the observed range before imputation.
- Added optional deterministic calendar/trend features (``add_time_features``, on by default) that stay observed through contiguous timestamp gaps.
- Training subsets that are identical across missingness patterns are fitted once and reused.
- Large speedup of the imputation pipeline (about 4.5x on the reference TimeSeriesImputer benchmark) with lower memory usage and unchanged imputation results: default FastRidge models are solved from incrementally accumulated Gram matrices, per-pattern training rows are found from an index of missing positions, normalization statistics and feature-selection scoring are computed from masked sums without full-matrix copies, and lag/lead features are built directly in a preallocated matrix.
- Fixed excessive memory retention caused by over-allocated NaN-position buffers.
- Removed dead internal helpers and the unused global NaN-position pipeline.

v0.2.2 (2026-02-11)
-------------------

- Added scikit-learn transformer compatibility for MultivariateImputer and TimeSeriesImputer.
- Ensured TimeSeriesImputer parameter updates propagate to its internal MultivariateImputer.
- Added pipeline-focused tests for both imputers.

v0.2.1 (2026-01-21)
-------------------

- Documentation overhaul with clearer parameters, expanded how-to content, and a new benchmarks page.
- New benchmark and utility scripts, plus runnable helpers for scripted runs.
- Expanded benchmark artifacts and example outputs (Titanic and PEMS-Bay assets).
- Imputation preprocessing updates (including pre-normalization adjustments).
- Estimator defaults updated (ridge intercept default behavior).
- Classifier baseline switched to DecisionTree.
- Reduced repeated concatenations to improve performance in core workflows.
- Timing tests to track performance regressions.
- Removed unused imports and refreshed plotting assets/styles used in docs.
