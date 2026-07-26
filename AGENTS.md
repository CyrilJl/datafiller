# Repository Guidelines

## Project Structure & Module Organization
- `src/datafiller/` contains the library code (src layout). Submodules include `estimators/`, `multivariate/`, `timeseries/`, and `datasets/`.
- `tests/` holds pytest suites (files named `test_*.py`).
- `scripts/` contains utility and benchmarking scripts; they regenerate the static assets in `docs/_static/` (see `scripts/README.md`).
- `docs/` hosts documentation sources and static assets.

## Algorithm Overview
- Rows to impute are grouped by their pattern of observed features; each pattern gets a model trained on the rows complete for those features (solved from a Gram matrix for the default `FastRidge` regressor, or via `estimator.fit` on the materialized subset otherwise). Per-pattern Gram matrices are assembled in `multivariate/_gram.py` from contributions cached per distinct NaN pattern of the training rows, in float64.
- `optimask` is a fallback heuristic used when fewer than `min_samples_train` complete rows exist: it searches the pareto front of row/column trade-offs for the largest NaN-free submatrix, preferring rectangles that keep at least `min_samples_train` rows (`min_rows=` parameter) and falling back to the unconstrained maximum-area choice when that is infeasible.
- Cells whose pattern never reaches `min_samples_train` training rows are filled by the `fallback` strategy (column mean / categorical mode by default, or left NaN with `fallback=None`).

## Build, Test, and Development Commands
- `pip install -e .` installs the package in editable mode for local development.
- `pytest` runs the full test suite.
- `pytest --cov=datafiller` runs tests with coverage (uses `pytest-cov` from `test` extras).
- `python scripts/run_scripts.py` is not provided; use `scripts/run_scripts.bat` or `scripts/run_scripts.sh` for scripted runs if needed.

## Coding Style & Naming Conventions
- Python code follows Ruff formatting rules with a line length of 120 (`pyproject.toml`).
- Use snake_case for functions and variables, PascalCase for classes, and `test_*.py` for test modules.
- Keep public APIs re-exported in `src/datafiller/__init__.py` consistent with module names.
- `pre-commit install` sets up Ruff format/lint hooks (`.pre-commit-config.yaml`); CI fails on unformatted or unlinted code.

## Testing Guidelines
- Testing uses `pytest` with optional coverage via `pytest-cov`.
- Name tests descriptively (e.g., `test_timeseries_imputer_handles_missing()`).
- Prefer unit tests in `tests/` over ad-hoc script validation.

## Commit & Pull Request Guidelines
- Recent commits use short, imperative summaries; some follow Conventional Commit style (e.g., `feat: ...`).
- Keep commit titles concise and scoped to a single change.
- PRs should include a brief description, testing notes (commands run), and links to relevant issues or documentation updates.

## Security & Configuration Tips
- Avoid committing generated artifacts like `.coverage`, caches, or large dataset files.
- If adding new datasets, place them under `src/datafiller/datasets/` and document their provenance.

## Discoveries & Lessons Log

Empirical findings about the library's behavior, recorded so design decisions stay traceable and future
work doesn't re-derive (or contradict) them. Append a dated entry when an experiment settles a design
question; keep the supporting scripts in `perf/` (gitignored) and cite them.

### 2026-07-19 — Calibrating `min_samples_train` (default 1 → 20)
- Sweep: 648 runs (6 datasets × 4 missingness patterns × 8 thresholds × 3 mask seeds), scripts
  `perf/min_samples_train_sweep.py` / `_analyze.py` / `_compare_objective.py`.
- The old default of 1 admitted ridge fits on 1–5 rows that were **worse than plain column means** in
  7/24 scenarios (all at ≥25% missingness on small data; worst 2.1× worse than mean). Below ~25%
  missingness the threshold never binds and its value is irrelevant.
- A fixed 20 was within 1.6% of the per-scenario-optimal threshold on average. **Both "smarter" rules
  lost to the constant**: fraction-of-rows fails on both ends (rounds to nothing on small n, rejects
  fine 200-row pools on large n) and per-feature `c·k` fails because the optimum tracks missingness
  intensity, not feature count.
- Measurement pitfall: strict thresholds skip the hardest cells, so error measured only on imputed
  cells is selection-biased. Score unimputed cells as mean-filled ("adjusted" metric) to compare
  thresholds fairly.
- The categorical path (DecisionTree) is insensitive to the threshold; the damage mechanism is
  unstable regression coefficients, which trees don't have.
- Higher thresholds cost no runtime — NaN-heavy runs get slightly faster (fewer fits).

### 2026-07-19 — Constrained optimask objective (`min_rows=`)
- optimask maximized *cells*, but the caller needs *rows*: a 15×40 rectangle (600 cells) beat a
  25×20 one (500 cells) and was then discarded because 15 < `min_samples_train` — the cells stayed
  NaN even though a usable training set existed on the very pareto front optimask had computed.
- Fix: masked argmax — maximize area subject to `rows ≥ min_rows`, unconstrained fallback when
  infeasible. Zero extra cost, byte-identical wherever the constraint doesn't bind.
- Effect (full re-run, same masks/seeds): coverage restored where the old objective collapsed
  (e.g. BreastCancer 25% MAR at t=200: 5% → 100% imputed; Wine 40% MAR at t=50: 4.7% → 99.2%),
  adjusted error never worse in any of 576 compared cells. The optimal default stays 20; overshooting
  it is now survivable. General lesson: **optimize what the caller needs, not a proxy**.

### 2026-07-19 — Deterministic nearest-feature selection
- `n_nearest_features` used to draw without replacement with probabilities proportional to the
  feature scores. That injects weaker predictors even though the library produces one imputation,
  not a multiple-imputation ensemble where sampling diversity would be useful.
- Evidence: `perf/nearest_feature_selection_eval.py` and the regenerated
  `docs/_static/multivariate_benchmark_results.csv`.
- Selecting the top scores instead improved every tested traffic scenario. On the published masks,
  nRMSE fell 2.5% / 33.7% for PEMS-BAY MAR / blocks and 10.4% / 19.4% for METR-LA. On independent
  MAR masks, PEMS-BAY improved 5.6% and METR-LA 7.3%.
- The change is at least as fast: paired independent-mask runs were 1.5% faster on PEMS-BAY and 3.9%
  faster on METR-LA. Sorting a few hundred scores is negligible compared with scoring and fitting,
  and avoids probability normalization plus random sampling. The tabular benchmark does not set
  `n_nearest_features`, so its models are unchanged.
- Final unclamped published-benchmark audit (identical masks): across all 30 numeric scenarios,
  geometric-mean nRMSE fell 2.5%; the four traffic scenarios that use `n_nearest_features` improved,
  the other 26 were identical, and none regressed.

### 2026-07-20 — Exploring alternatives to top-score nearest features
- Broad sweep: 570 screening runs across two traffic subsets and three wide/tabular datasets, using
  light MAR, heavy MAR, and block missingness. Compared deterministic top scores with uniform and
  score-weighted sampling, squared/square-root score temperatures, sampling from the top `2k`,
  50%/75% elite-plus-exploration hybrids, and deterministic rank-spread hybrids. Evidence:
  `perf/feature_selection_strategy_sweep.py` and `perf/analyze_feature_selection_strategy.py`.
- Deterministic top-score selection was the only strategy without a screening regression. The closest
  stochastic alternative, the 75%-elite hybrid, was 5.5% worse by geometric-mean nRMSE; squared-score
  sampling was 8.7% worse, ordinary score weighting 17.6% worse, and uniform sampling 61.9% worse.
  A deterministic 75%-elite rank spread came closest at 0.5% worse overall, but still had a 21.8%
  worst-case regression.
- Exploration sometimes helped light MAR on small data (the 50%-elite hybrid gained 4.5% there), but
  did not generalize. In 40 full-size PEMS-BAY/METR-LA runs covering MAR and three block-mask seeds,
  that hybrid was 30.1% worse geometrically and up to 2.28× worse; deterministic 75%-elite rank spread
  was 12.2% worse geometrically and up to 33.3% worse. Both were also worse across the two full MAR
  scenarios alone. Keep deterministic top-score selection as the single-imputation default; reserve
  randomized selection for an explicit ensemble/diversity use case rather than presumed accuracy.

### 2026-07-26 — Pipeline performance: pattern-Gram reuse and single-pass full-matrix work
- Measured against v0.3.2 on the same machine (baseline run from a `git worktree` at the previous
  commit, scenarios and peak working set in a throwaway harness): reference TimeSeriesImputer
  benchmark 0.99s → 0.21s (4.8x) with peak memory 1181 MB → 764 MB; 25k×25 tabular 2.04s → 0.25s
  (8.1x); all-columns traffic time series 7.23s → 1.83s (3.9x); wide 10k×250 all-columns
  270.9s → 23.7s (11.4x); wide 30k×250 all-columns with `n_nearest_features=35` 12.4s → 4.8s (2.6x).
  NaN-heavy 5k×60 at 25% missing is unchanged (41.5s → 40.1s) because it is dominated by `optimask`,
  which earlier work established is already near its per-call floor.
- **The dominant cost was redundant Gram accumulation, not the solves.** A training row whose NaNs
  sit in columns *N* is valid for *every* pattern excluding a superset of *N*, and the old code
  re-accumulated it once per such pattern: 2.25M row-outer-products (3.1 GFLOP) where 0.11 GFLOP of
  distinct information existed. Grouping training rows by exact NaN pattern and caching each group's
  Gram contribution cut that ~28x. Groups of one row get no cache (nothing to reuse) and a memory
  budget caps the rest; uncached rows are accumulated on demand, which is what makes the wide
  all-columns regime (nearly every row has a unique NaN pattern) degrade gracefully.
- **Finding candidate rows must not scan all rows.** Both the old `extra_rows_excluding` and a first
  version of the new kernel cost O(n_train) per pattern. Indexing rows and groups by their *lowest*
  NaN column and enumerating candidates from the pattern's excluded columns makes the cost follow the
  number of NaNs in those columns. Attributing each candidate to its lowest NaN column is what keeps
  the enumeration duplicate-free without a visited set.
- **Gram matrices need float64.** `sxx - outer(sx, sx) / n` cancels most of the Gram's magnitude, so
  float32 loses several digits on uncentred data. With `normalize=False` the old float32 path deviated
  from an exact float64 solve by up to 5.9% of a column's standard deviation; the new path is at
  2.8e-7. This was a real accuracy bug hidden by the fact that the default `normalize=True` centres
  the data and hides the cancellation.
- **Over half the original runtime was full-matrix NumPy passes, not modelling.** Only 0.41s of the
  1.28s benchmark was inside `_impute_col`; scoring (249 ms) and column statistics (207 ms) each made
  4-6 passes over a 211 MB matrix with 211 MB temporaries. Fusing them into single Numba traversals
  took them to 16 ms and 25 ms. Two specific traps: `bool_mask.sum(axis=0)` costs 47 ms where
  `np.count_nonzero(..., axis=0)` and `.all(axis=0)` cost 4 ms; and `pd.DataFrame(arr[:, cols])`
  spends 34 ms transposing 30 MB into pandas' column-major block, against 14 ms for a cache-blocked
  transpose handed over as a Fortran-ordered array (identical single-block frame).
- **Ownership, not extra passes, is what buys memory back.** Writing the standardized matrix and the
  original-scale output buffer from one read of the input, and rescaling imputed values as they are
  written, removed two full-matrix passes *and* the final denormalization. Standardizing in place when
  the caller owns the buffer (the private `_owns_input` flag `TimeSeriesImputer` passes, and any
  DataFrame input whose encoded matrix the imputer built itself) removed a further full-size array at
  no time cost — 209 MB of the 764 MB peak.
- Validation method: elementwise comparison against the pre-change outputs across 18 scenarios
  (numpy/pandas/mixed-categorical/timeseries/optimask-heavy/custom-regressor/degenerate columns), plus
  comparison of both old and new against a float64 materialized-ridge reference to establish *which*
  is closer to the exact answer when they differ. Differences that scale with column σ are the right
  metric; relative-to-value inflates cells that happen to sit near zero. Top-k nearest-feature
  selection was verified bit-identical, so score changes never moved which predictors were used.
