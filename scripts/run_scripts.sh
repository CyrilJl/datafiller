#!/usr/bin/env sh
set -e

python "scripts/generate_pems_timeseries_imputation_csv.py"
python "scripts/multivariate_benchmark.py"
python "scripts/render_titanic_tables.py"
