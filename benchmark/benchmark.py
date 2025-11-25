import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from datafiller import MultivariateImputer


def run_benchmark():
    """
    Compares the performance of MultivariateImputer and IterativeImputer
    using a shared list of underlying estimators.
    """
    # Load the California Housing dataset
    X_full, y_full = fetch_california_housing(return_X_y=True)
    # Use a subset of the data for a faster benchmark run
    X_full = X_full[::10]
    y_full = y_full[::10]
    n_samples, n_features = X_full.shape

    # Introduce missing values consistently
    rng = np.random.RandomState(0)
    X_missing = X_full.copy()
    missing_samples = np.arange(n_samples)
    missing_features = rng.choice(n_features, n_samples, replace=True)
    X_missing[missing_samples, missing_features] = np.nan

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
    ]

    results = []

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
                estimator=estimator_instance, max_iter=40, **iterative_imputer_params, random_state=0
            ),
        }

        for imputer_name, imputer in imputers_to_compare.items():
            start_time = time.time()
            if isinstance(imputer, MultivariateImputer):
                X_imputed = imputer(X_missing)
            else:
                X_imputed = imputer.fit_transform(X_missing)
            end_time = time.time()

            # Calculate MSE on the originally missing values
            mse = np.mean(
                (X_imputed[missing_samples, missing_features] - X_full[missing_samples, missing_features]) ** 2
            )

            results.append({"Imputer": imputer_name, "Time (s)": end_time - start_time, "MSE": mse})

    # --- Process and display results ---
    results_df = pd.DataFrame(results)
    print("\n--- Benchmark Results ---")
    print(results_df)

    # Save results to a CSV file
    results_df.to_csv("imputation_benchmark_results.csv", index=False)

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Imputer Performance Comparison", fontsize=16)

    # Time Comparison Plot
    results_df.plot(
        kind="barh",
        x="Imputer",
        y="Time (s)",
        ax=ax1,
        title="Execution Time",
        legend=False,
        color=["#1f77b4", "#ff7f0e"] * len(estimators),
    )
    ax1.set_xlabel("Time (s) (lower is better)")
    ax1.set_ylabel("")

    # MSE Comparison Plot
    results_df.plot(
        kind="barh",
        x="Imputer",
        y="MSE",
        ax=ax2,
        title="Imputation Error (MSE)",
        legend=False,
        color=["#1f77b4", "#ff7f0e"] * len(estimators),
    )
    ax2.set_xlabel("Mean Squared Error (lower is better)")
    ax2.set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("imputation_benchmark_comparison.png")
    print(
        "\nBenchmark complete. Results saved to 'imputation_benchmark_results.csv' and 'imputation_benchmark_comparison.png'."
    )


if __name__ == "__main__":
    run_benchmark()
