# Repository Guidelines

## Project Structure & Module Organization
- `datafiller/` contains the library code. Submodules include `estimators/`, `multivariate/`, `timeseries/`, and `datasets/`.
- `tests/` holds pytest suites (files named `test_*.py`).
- `scripts/` contains utility and benchmarking scripts.
- `docs/` hosts documentation sources and static assets.

## Algorithm Overview
- `optimask` is a heuristic that finds the largest submatrix without NaNs inside a matrix containing NaNs.
- Imputation then trains standard regression/classification models on that NaN-free subset and predicts missing values in the full data.

## Build, Test, and Development Commands
- `pip install -e .` installs the package in editable mode for local development.
- `pytest` runs the full test suite.
- `pytest --cov=datafiller` runs tests with coverage (uses `pytest-cov` from `test` extras).
- `python scripts/run_scripts.py` is not provided; use `scripts/run_scripts.bat` or `scripts/run_scripts.sh` for scripted runs if needed.

## Coding Style & Naming Conventions
- Python code follows Ruff formatting rules with a line length of 120 (`pyproject.toml`).
- Use snake_case for functions and variables, PascalCase for classes, and `test_*.py` for test modules.
- Keep public APIs re-exported in `datafiller/__init__.py` consistent with module names.

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
- If adding new datasets, place them under `datafiller/datasets/` and document their provenance.
