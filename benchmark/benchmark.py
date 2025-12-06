import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from datafiller import ELMImputer, MultivariateImputer
from datafiller.estimators.elm import ExtremeLearningMachine


def run_benchmark():
    """
    Compares the performance of MultivariateImputer and IterativeImputer
    using a shared list of underlying estimators.
    """
    datasets = [
        ("California Housing", fetch_california_housing),
        ("Diabetes", load_diabetes),
    ]

    results = []

    for dataset_name, fetch_func in datasets:
        print(f"\n===== Benchmarking on {dataset_name} dataset =====")
        # Load the dataset
        X_full, _ = fetch_func(return_X_y=True)

        # Use a subset of the data for a faster benchmark run
        if "Housing" in dataset_name:
            X_full = X_full[::10]

        # Scale data for better performance with some estimators
        scaler = StandardScaler()
        X_full = scaler.fit_transform(X_full)

        n_samples, n_features = X_full.shape

        # Introduce missing values consistently for each dataset
        rng = np.random.RandomState(0)
        X_missing = X_full.copy()
        missing_samples = np.arange(n_samples)
        missing_features = rng.choice(n_features, n_samples, replace=True)
        X_missing[missing_samples, missing_features] = np.nan

        # Warm-up for MultivariateImputer to account for Numba's JIT compilation
        print("Warming up MultivariateImputer (Numba JIT compilation)...")
        warmup_imputer = MultivariateImputer(estimator=BayesianRidge())
        # Use a small subset of data for a quick warm-up
        _ = warmup_imputer(X_missing[:5].copy())
        print("Warm-up complete.")

        # Define the list of estimators to be used for comparison
        estimators = [
            (
                "BayesianRidge",
                BayesianRidge(),
                {"tol": 1e-3},  # Tolerance for IterativeImputer
            ),
            (
                "RandomForest",
                RandomForestRegressor(
                    n_estimators=5,
                    max_depth=10,
                    bootstrap=True,
                    max_samples=0.5,
                    n_jobs=2,
                    random_state=0,
                ),
                {"tol": 1e-1},  # Tolerance for IterativeImputer
            ),
            (
                "GradientBoosting",
                GradientBoostingRegressor(n_estimators=10, random_state=0),
                {"tol": 1e-1},  # Tolerance for IterativeImputer
            ),
            ("ELM", ExtremeLearningMachine(random_state=0), {"tol": 1e-1}),
        ]

        # Iterate through each estimator and compare the two imputers
        for estimator_name, estimator_instance, iterative_imputer_params in estimators:
            print(f"--- Benchmarking with {estimator_name} ---")

            imputers_to_compare = {
                f"MultivariateImputer ({estimator_name})": MultivariateImputer(
                    estimator=estimator_instance.set_params(random_state=0)
                    if "random_state" in estimator_instance.get_params()
                    else estimator_instance
                ),
                f"IterativeImputer ({estimator_name})": IterativeImputer(
                    estimator=estimator_instance,
                    max_iter=40,
                    **iterative_imputer_params,
                    random_state=0,
                ),
            }

            if isinstance(estimator_instance, ExtremeLearningMachine):
                imputers_to_compare[f"ELMImputer ({estimator_name})"] = ELMImputer(
                    n_features=estimator_instance.n_features,
                    alpha=estimator_instance.alpha,
                    random_state=estimator_instance.random_state,
                )

            for imputer_name, imputer in imputers_to_compare.items():
                start_time = time.time()
                if isinstance(imputer, (MultivariateImputer, ELMImputer)):
                    X_imputed = imputer(X_missing)
                else:
                    X_imputed = imputer.fit_transform(X_missing)
                end_time = time.time()

                # Calculate metrics on the originally missing values
                y_true = X_full[missing_samples, missing_features]
                y_pred = X_imputed[missing_samples, missing_features]
                errors = y_pred - y_true

                mse = np.mean(errors**2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(errors))
                bias = np.mean(errors)

                # Calculate MAPE, avoiding division by zero
                non_zero_mask = y_true != 0
                mape = np.mean(np.abs(errors[non_zero_mask] / y_true[non_zero_mask])) * 100

                results.append(
                    {
                        "Dataset": dataset_name,
                        "Imputer": imputer_name,
                        "Time (s)": end_time - start_time,
                        "RMSE": rmse,
                        "MAE": mae,
                        "Bias": bias,
                        "MAPE (%)": mape,
                    }
                )

    # --- Process and display results ---
    results_df = pd.DataFrame(results)
    print("\n--- Benchmark Results ---")
    print(results_df)

    # Save results to a CSV file
    results_df.to_csv("benchmark/imputation_benchmark_results.csv", index=False)

    # Plot the results
    for dataset_name, group in results_df.groupby("Dataset"):
        metrics_to_plot = ["Time (s)", "RMSE", "MAE", "Bias", "MAPE (%)"]
        n_metrics = len(metrics_to_plot)
        n_imputers = len(group["Imputer"].unique())
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 2 + n_imputers * 0.4), sharey=True)
        fig.suptitle(f"Imputer Performance Comparison on {dataset_name}", fontsize=16)

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"] * (n_imputers // 3 + 1)

        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            group.plot(
                kind="barh",
                x="Imputer",
                y=metric,
                ax=ax,
                title=metric,
                legend=False,
                color=colors,
            )
            lower_is_better = metric not in ["Bias"]
            xlabel = f"{metric} ({'lower is better' if lower_is_better else 'closer to 0 is better'})"
            ax.set_xlabel(xlabel)
            ax.set_ylabel("")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = f"benchmark/imputation_benchmark_{dataset_name.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"Plot for {dataset_name} saved to '{filename}'")
    print("\nBenchmark complete. Results saved to 'benchmark/imputation_benchmark_results.csv'.")


if __name__ == "__main__":
    run_benchmark()
