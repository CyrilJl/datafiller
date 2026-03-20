"""Profile registered MultivariateImputer benchmark cases with cProfile and tracemalloc."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

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
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for data generation and the imputer.")
    parser.add_argument("--top", type=int, default=20, help="Number of profile and allocation rows to keep.")
    parser.add_argument("--sort", default="cumtime", help="pstats sort key.")
    parser.add_argument(
        "--output-dir",
        default="docs/_static/performance/profiles",
        help="Directory where profile summaries are written.",
    )
    return parser.parse_args()


def profile_case(case, seed: int, top: int, sort: str, output_dir: Path) -> dict[str, str | float]:
    from datafiller import MultivariateImputer
    from scripts._multivariate_perf_shared import clone_input_data, evaluate_imputation

    payload = case.builder(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Warm up Numba and the estimator before taking the profile.
    MultivariateImputer(rng=seed)(clone_input_data(payload.masked), **case.call_kwargs)

    imputer = MultivariateImputer(rng=seed)
    bench_input = clone_input_data(payload.masked)

    tracemalloc.start()
    profile = cProfile.Profile()
    started_at = time.perf_counter()
    profile.enable()
    imputed = imputer(bench_input, **case.call_kwargs)
    profile.disable()
    elapsed = time.perf_counter() - started_at
    _, peak_bytes = tracemalloc.get_traced_memory()
    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    metrics = evaluate_imputation(payload.truth, imputed, payload.mask)

    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream).sort_stats(sort)
    stats.print_stats(top)
    profile_text = stream.getvalue()

    allocations = snapshot.statistics("lineno")[:top]
    allocation_lines = [
        f"{stat.traceback.format()[-1].strip()} | size={stat.size / (1024 * 1024):.3f} MiB | count={stat.count}"
        for stat in allocations
    ]

    summary = {
        "git_branch": _git_value(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": _git_value(["git", "rev-parse", "--short", "HEAD"]),
        "benchmark": case.name,
        "description": case.description,
        "elapsed_seconds": elapsed,
        "peak_memory_mb": peak_bytes / (1024 * 1024),
        **metrics,
    }

    text_path = output_dir / f"{case.name}.txt"
    json_path = output_dir / f"{case.name}.json"
    text_path.write_text(
        "\n".join(
            [
                f"Benchmark: {case.name}",
                f"Description: {case.description}",
                f"Branch: {summary['git_branch']}",
                f"Commit: {summary['git_commit']}",
                f"Elapsed seconds: {elapsed:.6f}",
                f"Peak tracemalloc memory (MiB): {summary['peak_memory_mb']:.3f}",
                f"Coverage: {summary['coverage']}",
                f"RMSE: {summary['rmse']}",
                f"MAE: {summary['mae']}",
                f"Accuracy: {summary['accuracy']}",
                f"Remaining missing total: {summary['remaining_missing_total']}",
                "",
                "Top cProfile rows:",
                profile_text.rstrip(),
                "",
                "Top tracemalloc rows:",
                *allocation_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote profile summary to {text_path}")
    return {"text_path": str(text_path), "json_path": str(json_path), "elapsed_seconds": elapsed}


def main() -> int:
    from scripts._multivariate_perf_registry import get_benchmark_cases

    args = parse_args()
    cases = get_benchmark_cases(args.benchmark)
    output_dir = Path(args.output_dir)
    for case in cases:
        profile_case(case, seed=args.seed, top=args.top, sort=args.sort, output_dir=output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
