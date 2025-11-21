#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from scperturb_cmap.benchmarking import run_benchmark_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scPerturb-CMap benchmark suite.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset path (defaults to examples/data/benchmark_synthetic.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/benchmarks",
        help="Directory to write results and plots",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    metrics = run_benchmark_suite(out_dir, dataset_path=args.dataset)
    print(f"Wrote metrics to {out_dir} -> {metrics}")


if __name__ == "__main__":
    main()
