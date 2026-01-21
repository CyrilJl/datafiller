Changelog
=========

v0.2.1 (2026-01-21)
-------------------

Added
~~~~~
- New benchmark and utility scripts, plus runnable helpers for scripted runs.
- Expanded benchmark artifacts and example outputs (Titanic and PEMS-Bay assets).
- Timing tests to track performance regressions.

Changed
~~~~~~~
- Documentation overhaul with clearer parameters, expanded how-to content, and a new benchmarks page.
- Imputation preprocessing updates (including pre-normalization adjustments).
- Estimator defaults updated (ridge intercept default behavior).
- Classifier baseline switched to DecisionTree.
- Reduced repeated concatenations to improve performance in core workflows.

Fixed
~~~~~
- Removed unused imports and refreshed plotting assets/styles used in docs.
