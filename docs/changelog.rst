Changelog
=========

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
