from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px

from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.data.lincs_loader import load_lincs_long
from scperturb_cmap.io.schemas import TargetSignature


def _default_target() -> TargetSignature:
    return TargetSignature(genes=["G1", "G2", "G3"], weights=[1.0, 1.0, -1.0])


def load_benchmark_dataset(dataset_path: Optional[str] = None) -> Tuple[TargetSignature, pd.DataFrame, List[str]]:
    """Load a small benchmark dataset (or synthesize one)."""
    target = _default_target()
    positives = ["sig_inverter"]

    path = Path(dataset_path) if dataset_path else Path("examples/data/benchmark_synthetic.csv")
    if path.exists():
        if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
            df = pd.read_csv(path)
        else:
            df = load_lincs_long(str(path))
        return target, df, positives

    # Fallback synthetic table
    rows = []
    inv_scores = [-w for w in target.weights]
    conc_scores = [w for w in target.weights]
    noise = np.random.default_rng(0).normal(scale=0.1, size=len(target.weights))
    for gene, inv, conc, ns in zip(target.genes, inv_scores, conc_scores, noise):
        rows.append({"signature_id": "sig_inverter", "compound": "cmpd_inverter", "cell_line": "CL1", "gene_symbol": gene, "score": float(inv)})
        rows.append({"signature_id": "sig_concordant", "compound": "cmpd_concordant", "cell_line": "CL1", "gene_symbol": gene, "score": float(conc)})
        rows.append({"signature_id": "sig_noise", "compound": "cmpd_noise", "cell_line": "CL1", "gene_symbol": gene, "score": float(ns)})
    df = pd.DataFrame(rows)
    return target, df, positives


def recall_at_k(ranking_df: pd.DataFrame, positives: List[str], k: int) -> float:
    if ranking_df is None or ranking_df.empty:
        return 0.0
    top = ranking_df.head(k)["signature_id"].astype(str).tolist()
    return float(any(sig in positives for sig in top))


def precision_at_k(ranking_df: pd.DataFrame, positives: List[str], k: int) -> float:
    if ranking_df is None or ranking_df.empty:
        return 0.0
    top = ranking_df.head(k)["signature_id"].astype(str).tolist()
    hits = sum(1 for sig in top if sig in positives)
    return float(hits) / float(min(k, len(top)))


def evaluate_methods(
    target: TargetSignature,
    library_df: pd.DataFrame,
    positives: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compare scPerturb-CMap against a random baseline."""
    res = rank_drugs(target, library_df, method="baseline", top_k=50)
    ranking_df = res.ranking if isinstance(res.ranking, pd.DataFrame) else pd.DataFrame(res.ranking)
    k_values = [1, 3, 5]
    metrics = {
        "scperturb_baseline": {
            f"recall@{k}": recall_at_k(ranking_df, positives, k) for k in k_values
        }
    }

    shuffled = ranking_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    metrics["random"] = {f"recall@{k}": recall_at_k(shuffled, positives, k) for k in k_values}
    metrics["random"].update({f"precision@{k}": precision_at_k(shuffled, positives, k) for k in k_values})
    metrics["scperturb_baseline"].update({f"precision@{k}": precision_at_k(ranking_df, positives, k) for k in k_values})
    return metrics


def plot_metrics(metrics: Dict[str, Dict[str, float]], output_dir: Path) -> Path:
    records = []
    for method, vals in metrics.items():
        for metric, value in vals.items():
            records.append({"method": method, "metric": metric, "value": value})
    df = pd.DataFrame(records)
    fig = px.bar(df, x="metric", y="value", color="method", barmode="group", title="Benchmark metrics")
    fig.update_layout(
        xaxis_title="Metric",
        yaxis_title="Score",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "benchmark_metrics.html"
    try:
        import plotly.io as pio  # noqa: WPS433

        pio.write_html(fig, str(html_path), include_plotlyjs="cdn")
    except Exception:
        fig.write_html(str(html_path), include_plotlyjs="cdn")
    return html_path


def run_benchmark_suite(
    output_dir: Path,
    dataset_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Run a lightweight benchmark and write results/plots."""
    target, library_df, positives = load_benchmark_dataset(dataset_path)
    metrics = evaluate_methods(target, library_df, positives)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "benchmark_results.json"
    results_path.write_text(json.dumps(metrics, indent=2))
    plot_metrics(metrics, output_dir)
    return metrics
