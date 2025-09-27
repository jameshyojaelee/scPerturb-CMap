#!/usr/bin/env python3
"""
Fig 2: Recall@50 by cell line

Generates a grouped bar chart comparing Baseline vs MetricBlend recall@50 per cell line,
with 95% CI error bars, from an input CSV. Also writes a summary CSV of group statistics.

Requirements:
- pandas
- matplotlib (no seaborn)
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["cell_line", "method", "recall_at_50", "seed"]
REQUIRED_METHODS = {"Baseline", "MetricBlend"}


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Recall@50 by cell line comparing Baseline vs MetricBlend, with 95% CI error bars."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV with columns: cell_line, method, recall_at_50, seed",
    )
    parser.add_argument(
        "--output-png",
        default="figs/fig2_recall_by_cell_line.png",
        help="Output PNG path (default: figs/fig2_recall_by_cell_line.png)",
    )
    parser.add_argument(
        "--output-svg",
        default="figs/fig2_recall_by_cell_line.svg",
        help="Output SVG path (default: figs/fig2_recall_by_cell_line.svg)",
    )
    parser.add_argument(
        "--summary-csv",
        default="results/fig2_recall_summary.csv",
        help=(
            "Path to write summary table (mean, 95 CI, N seeds) "
            "(default: results/fig2_recall_summary.csv)"
        ),
    )
    return parser.parse_args(argv)


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    # Validate required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        msg = (
            f"Input CSV is missing required column(s): {', '.join(missing_cols)}. "
            f"Found columns: {list(df.columns)}"
        )
        raise SystemExit(msg)

    # Drop rows with NA in required fields
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    # Enforce required methods present and only those two
    methods_present = set(map(str, df["method"].unique()))
    if not REQUIRED_METHODS.issubset(methods_present):
        missing = REQUIRED_METHODS - methods_present
        raise SystemExit(
            "Input must contain both methods 'Baseline' and 'MetricBlend'. "
            f"Missing: {sorted(missing)}. Found: {sorted(methods_present)}"
        )
    extras = methods_present - REQUIRED_METHODS
    if extras:
        raise SystemExit(
            "Input must contain exactly two methods: 'Baseline' and 'MetricBlend'. "
            f"Unexpected methods present: {sorted(extras)}"
        )

    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    # Group by (cell_line, method), compute mean, std, n, and 95% CI
    grouped = (
        df.groupby(["cell_line", "method"], dropna=False)["recall_at_50"]
        .agg([("mean_recall_at_50", "mean"), ("std", "std"), ("n_seeds", "count")])
        .reset_index()
    )

    # Compute 95% CI using normal approximation; if n < 2, CI = 0
    def compute_ci(std: float, n: int) -> float:
        if n is None or n < 2 or not np.isfinite(std):
            return 0.0
        return float(1.96 * (std / math.sqrt(n)))

    grouped["ci95"] = [compute_ci(std, int(n)) for std, n in zip(grouped["std"], grouped["n_seeds"])]
    return grouped.drop(columns=["std"]).sort_values(["cell_line", "method"]).reset_index(drop=True)


def plot_grouped_bars(summary: pd.DataFrame, output_png: str, output_svg: str) -> None:
    # Keep only cell lines that have both methods present
    ct = summary.groupby("cell_line")["method"].nunique()
    ok_cells = sorted(ct[ct == 2].index.tolist())
    if not ok_cells:
        raise SystemExit(
            "No cell lines have both methods present; cannot create grouped bar chart."
        )

    # Pivot to obtain means and ci per method
    means_pivot = summary.pivot(index="cell_line", columns="method", values="mean_recall_at_50")
    ci_pivot = summary.pivot(index="cell_line", columns="method", values="ci95")

    means_pivot = means_pivot.loc[ok_cells]
    ci_pivot = ci_pivot.loc[ok_cells]

    x_labels = ok_cells
    baseline_vals = means_pivot["Baseline"].to_numpy()
    baseline_ci = ci_pivot["Baseline"].to_numpy()
    metric_vals = means_pivot["MetricBlend"].to_numpy()
    metric_ci = ci_pivot["MetricBlend"].to_numpy()

    n = len(x_labels)
    x = np.arange(n)
    width = 0.38

    # Figure width scaled by number of groups for readability
    fig_w = max(8.0, min(24.0, 0.6 * n + 4.0))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        yerr=baseline_ci,
        capsize=4,
        label="Baseline",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        metric_vals,
        width,
        yerr=metric_ci,
        capsize=4,
        label="MetricBlend",
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_ylabel("Recall@50")
    ax.set_xlabel("Cell line")
    ax.set_title("Fig 2. Recall@50 by cell line")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=(30 if n > 6 else 0), ha=("right" if n > 6 else "center"))
    ax.legend(title="Method")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()

    # Ensure output directories exist and save
    ensure_parent_dir(output_png)
    ensure_parent_dir(output_svg)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_svg, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    # Create default dirs if using defaults
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load and validate
    try:
        df = pd.read_csv(args.input)
    except Exception as exc:
        print(f"Failed to read input CSV: {exc}", file=sys.stderr)
        return 2

    try:
        df = validate_and_prepare(df)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2

    # Summarize
    summary = summarize(df)

    # Enforce that globally we have exactly the two methods
    present = set(summary["method"].unique())
    if present != REQUIRED_METHODS:
        print(
            "Input must contain exactly two methods: 'Baseline' and 'MetricBlend'. "
            f"Found: {sorted(present)}",
            file=sys.stderr,
        )
        return 2

    # Write summary CSV
    try:
        ensure_parent_dir(args.summary_csv)
        summary.to_csv(
            args.summary_csv,
            index=False,
            columns=["cell_line", "method", "mean_recall_at_50", "ci95", "n_seeds"],
        )
    except Exception as exc:
        print(f"Failed to write summary CSV: {exc}", file=sys.stderr)
        return 2

    # Plot
    try:
        plot_grouped_bars(summary, args.output_png, args.output_svg)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Failed to create plot: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


