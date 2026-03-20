"""Run registered MultivariateImputer benchmark cases."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="append", help="Benchmark case name. Can be passed multiple times.")
    parser.add_argument("--repeat", type=int, default=3, help="Timed repetitions per benchmark case.")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed warmup runs per benchmark case.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for data generation and the imputer.")
    parser.add_argument("--label", default=None, help="Optional label recorded in the output rows.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path. Parent directories are created automatically.",
    )
    return parser.parse_args()


def run_case(case, seed: int, repeat: int, warmup: int, label: str | None) -> list[dict]:
    from datafiller import MultivariateImputer
    from scripts._multivariate_perf_shared import clone_input_data, evaluate_imputation

    payload = case.builder(seed)

    for _ in range(warmup):
        imputer = MultivariateImputer(rng=seed)
        imputer(clone_input_data(payload.masked), **case.call_kwargs)

    rows: list[dict] = []
    for run_idx in range(repeat):
        imputer = MultivariateImputer(rng=seed)
        bench_input = clone_input_data(payload.masked)

        start = time.perf_counter()
        imputed = imputer(bench_input, **case.call_kwargs)
        elapsed = time.perf_counter() - start

        metrics = evaluate_imputation(payload.truth, imputed, payload.mask)
        rows.append(
            {
                "label": label or "",
                "git_branch": _git_value(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
                "git_commit": _git_value(["git", "rev-parse", "--short", "HEAD"]),
                "benchmark": case.name,
                "description": case.description,
                "dataset_kind": case.dataset_kind,
                "repeat_index": run_idx,
                "elapsed_seconds": elapsed,
                "call_kwargs": repr(case.call_kwargs),
                **payload.metadata,
                **metrics,
            }
        )

    return rows


def print_summary(results: pd.DataFrame) -> None:
    if results.empty:
        print("No benchmark results generated.")
        return

    summary = (
        results.groupby(["benchmark", "dataset_kind"], as_index=False)
        .agg(
            median_seconds=("elapsed_seconds", "median"),
            min_seconds=("elapsed_seconds", "min"),
            max_seconds=("elapsed_seconds", "max"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_coverage=("coverage", "mean"),
            remaining_missing_total=("remaining_missing_total", "mean"),
        )
        .sort_values("median_seconds")
    )
    print(summary.to_string(index=False))


def main() -> int:
    from scripts._multivariate_perf_registry import get_benchmark_cases

    args = parse_args()
    cases = get_benchmark_cases(args.benchmark)

    rows: list[dict] = []
    for case in cases:
        print(f"Running {case.name} ({case.description})")
        rows.extend(run_case(case, seed=args.seed, repeat=args.repeat, warmup=args.warmup, label=args.label))

    results = pd.DataFrame(rows)
    print_summary(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Saved benchmark results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
