"""Registry of repeatable MultivariateImputer performance cases."""

from __future__ import annotations

from scripts._multivariate_perf_shared import BenchmarkCase, build_mixed_dataframe_payload, build_numeric_payload


BENCHMARK_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        name="numeric_mar_medium",
        description="5000x25 float32 array with 8% MAR missingness.",
        dataset_kind="array",
        builder=lambda seed: build_numeric_payload(
            seed=seed,
            name="numeric_mar_medium",
            rows=5000,
            cols=25,
            mask_kind="mar",
            missing_ratio=0.08,
        ),
    ),
    BenchmarkCase(
        name="numeric_block_medium",
        description="5000x25 float32 array with contiguous missing blocks on 25% of columns.",
        dataset_kind="array",
        builder=lambda seed: build_numeric_payload(
            seed=seed,
            name="numeric_block_medium",
            rows=5000,
            cols=25,
            mask_kind="block",
            frac_columns=0.25,
            block_length_ratio=0.18,
        ),
    ),
    BenchmarkCase(
        name="numeric_feature_selection",
        description="3500x80 float32 array with 8% MAR and sampled nearest features.",
        dataset_kind="array",
        builder=lambda seed: build_numeric_payload(
            seed=seed,
            name="numeric_feature_selection",
            rows=3500,
            cols=80,
            mask_kind="mar",
            missing_ratio=0.08,
        ),
        call_kwargs={"n_nearest_features": 12},
    ),
    BenchmarkCase(
        name="numeric_wide_light_missing",
        description="2500x120 float32 array with 4% MAR missingness.",
        dataset_kind="array",
        builder=lambda seed: build_numeric_payload(
            seed=seed,
            name="numeric_wide_light_missing",
            rows=2500,
            cols=120,
            mask_kind="mar",
            missing_ratio=0.04,
        ),
    ),
    BenchmarkCase(
        name="mixed_dataframe_mar",
        description="4000-row mixed dataframe with numeric, categorical, string, and boolean columns.",
        dataset_kind="dataframe",
        builder=lambda seed: build_mixed_dataframe_payload(
            seed=seed,
            name="mixed_dataframe_mar",
            rows=4000,
            missing_ratio=0.10,
        ),
    ),
)


def get_benchmark_cases(names: list[str] | None = None) -> list[BenchmarkCase]:
    """Return registered benchmark cases, optionally filtered by name."""
    registry = {case.name: case for case in BENCHMARK_CASES}
    if names is None:
        return list(BENCHMARK_CASES)

    missing = [name for name in names if name not in registry]
    if missing:
        raise KeyError(f"Unknown benchmark case(s): {missing}")
    return [registry[name] for name in names]
