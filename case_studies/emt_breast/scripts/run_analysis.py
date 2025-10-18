#!/usr/bin/env python3
"""
Workflow runner for the EMT breast cancer case study.

Provides a single entry point that runs target QC, baseline scoring, and optional
metric blending so the documentation can reference a reproducible command.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.data.lincs_loader import load_lincs_long
from scperturb_cmap.data.signatures import summarize_target_signature
from scperturb_cmap.io.schemas import TargetSignature

app = typer.Typer(add_completion=False)

CASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = CASE_DIR / "data" / "target.json"
DEFAULT_LIBRARY = Path(__file__).resolve().parents[3] / "examples" / "data" / "lincs_demo.parquet"
DEFAULT_RESULTS = CASE_DIR / "results"
DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "workspace" / "artifacts" / "best.pt"


def _load_target(path: Path) -> TargetSignature:
    if not path.exists():
        raise typer.BadParameter(f"Target JSON not found at {path}")
    return TargetSignature.model_validate_json(path.read_text())


def _load_library(path: Path):
    if not path.exists():
        raise typer.BadParameter(f"LINCS library not found at {path}")
    return load_lincs_long(str(path))


@app.command()
def main(
    target_json: Path = typer.Option(
        DEFAULT_TARGET,
        help="Path to TargetSignature JSON produced by `scperturb-cmap make-target`.",
    ),
    library: Path = typer.Option(
        DEFAULT_LIBRARY,
        help="Long-form LINCS table (Parquet/CSV/TSV).",
    ),
    results_dir: Path = typer.Option(
        DEFAULT_RESULTS,
        help="Directory to write scoring outputs.",
    ),
    metric_model: Optional[Path] = typer.Option(
        DEFAULT_MODEL,
        help="Optional DualEncoder checkpoint for metric blending.",
    ),
    top_k: int = typer.Option(100, help="Number of rows to keep in each ranking."),
) -> None:
    """Run baseline (and optional metric) scoring for the case study target."""
    typer.echo(f"[scPerturb-CMap] loading target from {target_json}")
    target = _load_target(target_json)

    typer.echo(f"[scPerturb-CMap] loading LINCS library from {library}")
    library_df = _load_library(library)

    results_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("[scPerturb-CMap] computing target summary")
    summary = summarize_target_signature(
        target, library_genes=library_df["gene_symbol"].unique()
    )
    (results_dir / "target_summary.json").write_text(json.dumps(summary, indent=2))

    typer.echo("[scPerturb-CMap] scoring baseline ensemble")
    baseline = rank_drugs(
        target_signature=target,
        library=library_df,
        method="baseline",
        top_k=top_k,
    )
    baseline_path = results_dir / "baseline_rankings.parquet"
    baseline.ranking.to_parquet(baseline_path, index=False)
    typer.echo(f"[scPerturb-CMap] wrote baseline results to {baseline_path}")

    if metric_model:
        if metric_model.exists():
            typer.echo(f"[scPerturb-CMap] scoring metric blend using {metric_model}")
            metric = rank_drugs(
                target_signature=target,
                library=library_df,
                method="metric",
                model_path=str(metric_model),
                top_k=top_k,
                auto_blend=True,
            )
            metric_path = results_dir / "metric_rankings.parquet"
            metric.ranking.to_parquet(metric_path, index=False)
            typer.echo(f"[scPerturb-CMap] wrote metric results to {metric_path}")
        else:
            typer.echo(
                f"[scPerturb-CMap] metric model {metric_model} not found; skipping metric scoring",
                err=True,
            )

    typer.echo("[scPerturb-CMap] analysis complete")


if __name__ == "__main__":
    app()

