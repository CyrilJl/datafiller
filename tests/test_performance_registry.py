from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._multivariate_perf_registry import BENCHMARK_CASES, get_benchmark_cases
from scripts._multivariate_perf_shared import evaluate_imputation


def test_benchmark_registry_names_are_unique():
    names = [case.name for case in BENCHMARK_CASES]
    assert len(names) == len(set(names))


def test_benchmark_registry_payloads_are_masked_consistently():
    for case in get_benchmark_cases():
        payload = case.builder(0)
        assert payload.mask.shape == payload.truth.shape
        assert payload.mask.shape == payload.masked.shape

        if isinstance(payload.truth, pd.DataFrame):
            for col_idx, col in enumerate(payload.truth.columns):
                masked_values = payload.masked.loc[payload.mask[:, col_idx], col]
                assert masked_values.isna().all()
        else:
            assert np.isnan(payload.masked[payload.mask]).all()


def test_evaluate_imputation_returns_perfect_scores_for_ground_truth():
    for case in get_benchmark_cases():
        payload = case.builder(0)
        metrics = evaluate_imputation(payload.truth, payload.truth, payload.mask)
        assert metrics["remaining_missing_total"] == 0
        assert metrics["coverage"] == 1.0
        if not np.isnan(metrics["rmse"]):
            assert metrics["rmse"] == 0.0
            assert metrics["mae"] == 0.0
        if not np.isnan(metrics["accuracy"]):
            assert metrics["accuracy"] == 1.0
