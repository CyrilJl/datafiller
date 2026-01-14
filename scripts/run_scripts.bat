@echo off
setlocal

python "scripts\generate_pems_timeseries_imputation_csv.py"
if errorlevel 1 exit /b 1

python "scripts\multivariate_benchmark.py"
if errorlevel 1 exit /b 1

python "scripts\render_titanic_tables.py"
if errorlevel 1 exit /b 1
