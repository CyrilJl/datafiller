# Repository Guidelines

## Project Structure & Module Organization
- `src/datafiller/` contains the library code (src layout). Submodules include `estimators/`, `multivariate/`, `timeseries/`, and `datasets/`.
- `tests/` holds pytest suites (files named `test_*.py`).
- `scripts/` contains utility and benchmarking scripts; they regenerate the static assets in `docs/_static/` (see `scripts/README.md`).
- `docs/` hosts documentation sources and static assets.

## Algorithm Overview
- Rows to impute are grouped by their pattern of observed features; each pattern gets a model trained on the rows complete for those features (solved from an incrementally accumulated Gram matrix for the default `FastRidge` regressor, or via `estimator.fit` on the materialized subset otherwise).
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
