from __future__ import annotations

import json
import logging
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import typer

from scperturb_cmap import __version__
from scperturb_cmap.analysis.aggregate import collapse_replicates_modz
from scperturb_cmap.analysis.power import (
    bootstrap_rank_confidence,
    compute_signature_stability,
    estimate_signature_sample_size,
    permutation_significance_test,
    recommend_min_cells_per_cluster,
    simulate_false_discovery_rate,
)
from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.data.lincs_gctx import gctx_to_long
from scperturb_cmap.data.lincs_loader import load_lincs_long
from scperturb_cmap.data.preprocess import harmonize_symbols
from scperturb_cmap.data.resources import derive_landmarks_from_gene_info, load_l1000_landmarks
from scperturb_cmap.data.scrna_loader import load_h5ad
from scperturb_cmap.data.signatures import (
    summarize_target_signature,
    target_from_cluster,
    target_from_gene_lists,
)
from scperturb_cmap.io.schemas import TargetSignature
from scperturb_cmap.utils.device import get_device

app = typer.Typer(name="scperturb-cmap", help="scPerturb-CMap command line interface")
power_app = typer.Typer(name="power", help="Power analysis utilities")
app.add_typer(power_app, name="power")


def _load_table(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise typer.BadParameter(f"File not found: {path}")

    suffix = file_path.suffix.lower()
    try:
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(file_path, engine="pyarrow")
        if suffix in {".tsv", ".txt"}:
            return pd.read_csv(file_path, sep="\t")
        if suffix in {".json", ".jsonl"}:
            return pd.read_json(file_path)
        return pd.read_csv(file_path)
    except Exception as exc:  # pragma: no cover - pandas error message forwarded
        raise typer.BadParameter(f"Failed to load table '{path}': {exc}") from exc


def _parse_int_list(text: Optional[str]) -> Optional[List[int]]:
    if text is None:
        return None
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError as exc:  # pragma: no cover - user input path
        raise typer.BadParameter(f"Could not parse integer list from '{text}'") from exc


def _write_dataframe(df: pd.DataFrame, path: Optional[str]) -> None:
    if path is None:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)


@app.callback()
def _configure(
    ctx: typer.Context,
    log_level: str = typer.Option(
        "INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
        case_sensitive=False,
    ),
) -> None:
    """Configure global logging for the CLI."""
    lvl = getattr(logging, str(log_level).upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="[%(levelname)s] %(message)s")


@power_app.command("sample-size")
def power_sample_size(
    h5ad: str = typer.Option(..., help="Path to .h5ad file"),
    cluster_key: str = typer.Option(..., help="Observation column for cluster labels"),
    cluster: str = typer.Option(..., help="Cluster to evaluate"),
    reference: str = typer.Option(
        "rest", help="Reference label or 'rest' to use all other cells"
    ),
    sample_sizes: Optional[str] = typer.Option(
        None, help="Optional comma-separated list of sample sizes"
    ),
    replicates: int = typer.Option(50, help="Bootstrap replicates per sample size"),
    method: str = typer.Option(
        "rank_biserial", help="Differential signal method for signature construction"
    ),
    pseudobulk_key: Optional[str] = typer.Option(
        None, help="Optional obs column for pseudobulk aggregation"
    ),
    correlation_metric: str = typer.Option(
        "spearman", help="Correlation metric for comparing signatures"
    ),
    threshold: float = typer.Option(
        0.7, help="Correlation threshold indicating sufficient stability"
    ),
    random_seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
    summary_output: Optional[str] = typer.Option(
        None, help="Optional CSV/Parquet path for summary table"
    ),
    history_output: Optional[str] = typer.Option(
        None, help="Optional CSV/Parquet path for bootstrap history"
    ),
    json_output: Optional[str] = typer.Option(
        None, help="Optional JSON output path with combined results"
    ),
) -> None:
    """Estimate minimum cells required for a stable target signature."""

    adata = load_h5ad(h5ad)
    sizes = _parse_int_list(sample_sizes)
    result = estimate_signature_sample_size(
        adata,
        cluster_key=cluster_key,
        cluster=cluster,
        reference=reference,
        sample_sizes=sizes,
        replicates=replicates,
        method=method,
        pseudobulk_key=pseudobulk_key,
        correlation_metric=correlation_metric,
        threshold=threshold,
        random_state=random_seed,
    )

    baseline_msg = (
        f"Baseline cells: {result.baseline_cells}; "
        f"recommended sample size: {result.recommended_size} "
        f"(threshold={result.threshold})"
    )
    typer.echo(baseline_msg)
    typer.echo(result.summary.to_string(index=False))

    _write_dataframe(result.summary, summary_output)
    _write_dataframe(result.history, history_output)

    if json_output:
        payload: Dict[str, Any] = {
            "recommended_size": result.recommended_size,
            "threshold": result.threshold,
            "baseline_cells": result.baseline_cells,
            "summary": result.summary.to_dict(orient="records"),
            "history": result.history.to_dict(orient="records"),
        }
        out_path = Path(json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))


@power_app.command("min-cells")
def power_min_cells(
    h5ad: str = typer.Option(..., help="Path to .h5ad file"),
    cluster_key: str = typer.Option(..., help="Observation column for clusters"),
    reference: str = typer.Option(
        "rest", help="Reference cluster label or 'rest'"
    ),
    sample_sizes: Optional[str] = typer.Option(
        None, help="Optional comma-separated list of sample sizes to evaluate"
    ),
    replicates: int = typer.Option(50, help="Bootstrap replicates per sample size"),
    threshold: float = typer.Option(
        0.7, help="Correlation threshold indicating sufficient stability"
    ),
    method: str = typer.Option(
        "rank_biserial", help="Differential signal method for signature construction"
    ),
    pseudobulk_key: Optional[str] = typer.Option(
        None, help="Optional obs column for pseudobulk aggregation"
    ),
    correlation_metric: str = typer.Option(
        "spearman", help="Correlation metric for comparing signatures"
    ),
    random_seed: Optional[int] = typer.Option(None, help="Random seed"),
    recommendations_output: Optional[str] = typer.Option(
        None, help="Optional CSV/Parquet path for per-cluster recommendations"
    ),
    summaries_dir: Optional[str] = typer.Option(
        None, help="Optional directory to write per-cluster summary tables"
    ),
    histories_dir: Optional[str] = typer.Option(
        None, help="Optional directory to write per-cluster bootstrap histories"
    ),
    json_output: Optional[str] = typer.Option(
        None, help="Optional JSON output path with all results"
    ),
) -> None:
    """Recommend minimum cells per cluster to reach the desired stability."""

    adata = load_h5ad(h5ad)
    sizes = _parse_int_list(sample_sizes)
    result = recommend_min_cells_per_cluster(
        adata,
        cluster_key=cluster_key,
        reference=reference,
        sample_sizes=sizes,
        replicates=replicates,
        threshold=threshold,
        method=method,
        pseudobulk_key=pseudobulk_key,
        correlation_metric=correlation_metric,
        random_state=random_seed,
    )

    recommendations = result["recommendations"].copy()
    typer.echo(recommendations.to_string(index=False))
    _write_dataframe(recommendations, recommendations_output)

    if summaries_dir:
        summaries_path = Path(summaries_dir)
        summaries_path.mkdir(parents=True, exist_ok=True)
        for cluster, summary_df in result["summaries"].items():
            _write_dataframe(summary_df, str(summaries_path / f"summary_{cluster}.csv"))

    if histories_dir:
        histories_path = Path(histories_dir)
        histories_path.mkdir(parents=True, exist_ok=True)
        for cluster, history_df in result["histories"].items():
            _write_dataframe(history_df, str(histories_path / f"history_{cluster}.csv"))

    if json_output:
        payload = {
            "recommendations": recommendations.to_dict(orient="records"),
            "summaries": {
                cluster: df.to_dict(orient="records")
                for cluster, df in result["summaries"].items()
            },
            "histories": {
                cluster: df.to_dict(orient="records")
                for cluster, df in result["histories"].items()
            },
        }
        out_path = Path(json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))


@power_app.command("rank-ci")
def power_rank_ci(
    table: str = typer.Option(..., help="Path to replicate-level score table"),
    id_col: str = typer.Option("signature_id", help="Column with signature identifiers"),
    score_col: str = typer.Option("score", help="Column with score values"),
    replicate_col: str = typer.Option(
        "replicate_id", help="Column with replicate identifiers"
    ),
    n_boot: int = typer.Option(1000, help="Number of bootstrap resamples"),
    ci: float = typer.Option(0.95, help="Confidence interval level"),
    aggfunc: str = typer.Option(
        "mean", help="Aggregation over replicates (mean|median|sum)"
    ),
    ascending: bool = typer.Option(
        True, help="Sort ascending when ranking scores (lower implies better)"
    ),
    random_seed: Optional[int] = typer.Option(None, help="Random seed"),
    output: Optional[str] = typer.Option(
        None, help="Optional CSV/Parquet path for bootstrap rank summary"
    ),
    json_output: Optional[str] = typer.Option(
        None, help="Optional JSON output path"
    ),
) -> None:
    """Bootstrap ranking confidence intervals from replicate-level scores."""

    df = _load_table(table)
    res = bootstrap_rank_confidence(
        df,
        id_col=id_col,
        score_col=score_col,
        replicate_col=replicate_col,
        n_boot=n_boot,
        ci=ci,
        aggfunc=aggfunc,
        ascending=ascending,
        random_state=random_seed,
    )
    typer.echo(res.to_string(index=False))
    _write_dataframe(res, output)

    if json_output:
        out_path = Path(json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res.to_dict(orient="records"), indent=2))


@power_app.command("stability")
def power_stability(
    table: str = typer.Option(..., help="Path to replicate-level gene score table"),
    signature_col: str = typer.Option(
        "signature_id", help="Column containing signature identifiers"
    ),
    replicate_col: str = typer.Option(
        "replicate_id", help="Column containing replicate identifiers"
    ),
    gene_col: str = typer.Option("gene_symbol", help="Column with gene IDs"),
    score_col: str = typer.Option("score", help="Column with score values"),
    method: str = typer.Option(
        "spearman", help="Pairwise correlation metric (pearson|spearman|kendall|cosine)"
    ),
    output: Optional[str] = typer.Option(
        None, help="Optional CSV/Parquet path for stability metrics"
    ),
    json_output: Optional[str] = typer.Option(
        None, help="Optional JSON output path"
    ),
) -> None:
    """Compute per-signature stability metrics across replicates."""

    df = _load_table(table)
    res = compute_signature_stability(
        df,
        signature_col=signature_col,
        replicate_col=replicate_col,
        gene_col=gene_col,
        score_col=score_col,
        method=method,
    )
    typer.echo(res.to_string(index=False))
    _write_dataframe(res, output)
    if json_output:
        out_path = Path(json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res.to_dict(orient="records"), indent=2))


@power_app.command("fdr")
def power_fdr(
    table: str = typer.Option(..., help="Path to ranked table with hit labels"),
    score_col: str = typer.Option("score", help="Column containing ranking scores"),
    label_col: str = typer.Option("is_hit", help="Boolean/int column flagging known hits"),
    top_k: int = typer.Option(50, help="Number of top rows to evaluate"),
    n_sim: int = typer.Option(1000, help="Number of label permutations"),
    descending: bool = typer.Option(
        False, help="Set if higher scores are more significant"
    ),
    random_seed: Optional[int] = typer.Option(None, help="Random seed"),
    json_output: Optional[str] = typer.Option(
        None, help="Optional JSON output path"
    ),
) -> None:
    """Simulate expected false discoveries using label permutations."""

    df = _load_table(table)
    res = simulate_false_discovery_rate(
        df,
        score_col=score_col,
        label_col=label_col,
        top_k=top_k,
        n_sim=n_sim,
        ascending=not descending,
        random_state=random_seed,
    )
    typer.echo(
        " | ".join(
            [
                f"top_k={res['top_k']}",
                f"observed_hits={res['observed_hits']}",
                f"expected_false={res['expected_false_positives']:.2f}",
                f"estimated_fdr={res['estimated_fdr']:.3f}",
                f"p_value={res['p_value']:.4f}",
            ]
        )
    )

    if json_output:
        payload = {
            **{k: v for k, v in res.items() if k != "null_distribution"},
            "null_distribution": res["null_distribution"].tolist(),
        }
        out_path = Path(json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))


@power_app.command("permutation-test")
def power_permutation_test(
    group_a: List[float] = typer.Option(
        ..., "--group-a", help="Observations from group A", show_default=False
    ),
    group_b: List[float] = typer.Option(
        ..., "--group-b", help="Observations from group B", show_default=False
    ),
    statistic: str = typer.Option(
        "difference_in_means",
        help="Statistic to compare groups (difference_in_means|difference_in_medians|cohens_d)",
    ),
    n_permutations: int = typer.Option(1000, help="Number of permutations"),
    alternative: str = typer.Option(
        "two-sided", help="two-sided|greater|less alternative hypothesis"
    ),
    random_seed: Optional[int] = typer.Option(None, help="Random seed"),
    json_output: Optional[str] = typer.Option(
        None, help="Optional JSON output path"
    ),
) -> None:
    """Permutation-based significance test for two groups."""

    if not group_a or not group_b:
        raise typer.BadParameter("Both --group-a and --group-b require at least one value")

    res = permutation_significance_test(
        group_a,
        group_b,
        statistic=statistic,
        n_permutations=n_permutations,
        random_state=random_seed,
        alternative=alternative,
    )
    stats_msg = (
        f"statistic={res['observed_statistic']:.4f} | "
        f"p_value={res['p_value']:.4f} | "
        f"statistic_name={res['statistic']}"
    )
    typer.echo(stats_msg)

    if json_output:
        payload = {
            "observed_statistic": res["observed_statistic"],
            "p_value": res["p_value"],
            "statistic": res["statistic"],
            "null_distribution": res["null_distribution"].tolist(),
        }
        out_path = Path(json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
@app.command("validate-h5ad")
def validate_h5ad(
    h5ad: str = typer.Option(..., help="Path to .h5ad to validate"),
    expect_genes: Optional[str] = typer.Option(
        None, help="'L1000' or path to newline-delimited gene list"
    ),
    backed: bool = typer.Option(False, help="Use AnnData backed mode for large files"),
) -> None:
    """Validate .h5ad gene symbols and overlap with expected library genes."""
    import re

    import anndata as ad

    adata = ad.read_h5ad(h5ad, backed="r" if backed else None)
    var = adata.var
    # Determine symbol column
    symbol_col = None
    for c in ["gene_symbol", "feature_name", "SYMBOL", "symbol", "gene" ]:
        if c in var.columns:
            symbol_col = c
            break
    symbols = []
    if symbol_col is not None:
        vals = var[symbol_col].astype(str).tolist()
        # strip Ensembl-like version suffixes embedded in names
        symbols = [re.sub(r"\..*$", "", s).strip().upper() for s in vals]
    elif "feature_id" in var.columns:
        vals = var["feature_id"].astype(str).tolist()
        symbols = [re.sub(r"\..*$", "", s).strip().upper() for s in vals]
    else:
        # fallback to var_names
        symbols = [str(g).strip().upper() for g in list(adata.var_names)]

    n_vars = int(adata.n_vars)
    coverage = int(sum(bool(s) for s in symbols))
    typer.echo(f"genes: {n_vars:,}; symbol coverage: {coverage:,} ({coverage/max(1,n_vars):.1%})")

    expected = None
    if expect_genes:
        if expect_genes.upper() == "L1000":
            expected = set(load_l1000_landmarks(None))
        else:
            from pathlib import Path
            expected = {
                line.strip().upper()
                for line in Path(expect_genes).read_text().splitlines()
                if line.strip()
            }
    if expected is not None:
        symset = set(symbols)
        overlap = len(symset & set(map(str.upper, expected)))
        typer.echo(f"overlap with expected genes: {overlap:,}")
        missing = sorted(list(set(expected) - symset))[:10]
        if missing:
            typer.echo(f"example missing: {missing}")

    # obs hints
    obs_cols = list(adata.obs.columns)
    cluster_keys = [k for k in obs_cols if k.lower() in {"leiden","louvain","cluster","clusters"}]
    pseudo_keys = [
        k for k in obs_cols if any(s in k.lower() for s in ["sample", "donor", "patient", "dataset_id"])  # noqa: E501
    ]
    typer.echo(f"cluster candidates: {cluster_keys}")
    typer.echo(f"pseudobulk candidates: {pseudo_keys}")



@app.command()
def version() -> None:
    """Print package version."""
    typer.echo(__version__)


@app.command()
def device() -> None:
    """Print the selected compute device (cuda|mps|cpu)."""
    typer.echo(get_device())


@app.command()
def diagnose() -> None:
    """Print environment diagnostics as JSON."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": get_device(),
        "torch": getattr(torch, "__version__", "NA"),
        "pandas": getattr(pd, "__version__", "NA"),
    }
    print(json.dumps(info, indent=2))


@app.command("prepare-lincs")
def prepare_lincs(
    input: Optional[str] = typer.Option(
        None,
        help="Input LINCS long file (csv/tsv/parquet)",
    ),
    output: str = typer.Option(
        "examples/data/lincs_demo.parquet",
        help="Output Parquet path",
    ),
    genes_file: Optional[str] = typer.Option(
        None,
        help="Optional text file of gene symbols to keep",
    ),
    landmarks: bool = typer.Option(
        False,
        help="Filter to the L1000 landmark genes",
    ),
    landmarks_file: Optional[str] = typer.Option(
        None,
        help="Optional override path to L1000 landmark list",
    ),
    gctx: Optional[str] = typer.Option(
        None,
        help="Optional Level 5 GCTX to convert to long format",
    ),
    gene_info: Optional[str] = typer.Option(
        None,
        help="Optional gene_info table for mapping IDs to symbols",
    ),
    sig_info: Optional[str] = typer.Option(
        None,
        help="Optional sig_info table for metadata (sig_id, cell_id, pert_iname, etc.)",
    ),
    inst_info: Optional[str] = typer.Option(
        None,
        help="Optional inst_info table for additional metadata (not required)",
    ),
    repurposing: Optional[str] = typer.Option(
        None,
        help="Optional Repurposing Hub annotations (for MOA/targets)",
    ),
    pert_type: Optional[str] = typer.Option(
        None,
        help="Optional perturbation type filter (e.g., TRT_CP)",
    ),
    chunk_cols: int = typer.Option(
        0,
        help="If >0, write GCTX conversion in column chunks",
    ),
    partition_by: Optional[str] = typer.Option(
        None,
        help="Optional column to partition Parquet dataset by (e.g., cell_line)",
    ),
) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # If GCTX provided, convert using ingestion path
    if gctx:
        lm: Optional[list[str]] = None
        if landmarks:
            if landmarks_file:
                lm = load_l1000_landmarks(landmarks_file)
            elif gene_info and Path(gene_info).exists():
                lm = derive_landmarks_from_gene_info(gene_info)
            else:
                lm = load_l1000_landmarks(None)
        df = gctx_to_long(
            gctx,
            gene_info_path=gene_info,
            repurposing_path=repurposing,
            sig_info_path=sig_info,
            inst_info_path=inst_info,
            landmarks=lm,
            pert_type=pert_type,
            out_path=str(out),
            chunk_cols=int(chunk_cols) if chunk_cols and chunk_cols > 0 else 0,
            partition_by=partition_by,
        )
        # If chunked write was used, df will be empty; emit a summary by reading the output head
        if df.empty and out.exists():
            # Best-effort summary; partitioned datasets may fail to load due to mixed
            # dictionary/null encodings across chunks. In that case, just report path.
            try:
                df = pd.read_parquet(out, engine="pyarrow")
            except Exception:
                df = pd.DataFrame()
        if not df.empty:
            typer.echo(
                (
                    "Wrote "
                    f"{len(df):,} rows, "
                    f"{df['signature_id'].nunique():,} signatures, "
                    f"{df['gene_symbol'].nunique():,} genes -> {out}"
                )
            )
        else:
            typer.echo(f"Wrote -> {out}")
        return

    # Otherwise, treat input as an existing long-form library
    if not input:
        raise typer.BadParameter("Provide either --gctx or --input pointing to a long-form table")
    df = load_lincs_long(input)

    # Gene filters
    gene_list: list[str] = []
    if genes_file:
        gene_list = [
            line.strip()
            for line in Path(genes_file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    elif landmarks:
        gene_list = load_l1000_landmarks(landmarks_file)
    if gene_list:
        genes = harmonize_symbols(gene_list)
        df = df.assign(gsym=df["gene_symbol"].astype(str).str.strip().str.upper())
        df = df[df["gsym"].isin(set(genes))].drop(columns=["gsym"]).reset_index(drop=True)

    df.to_parquet(out, engine="pyarrow", index=False)
    typer.echo(
        f"Wrote {len(df):,} rows, {df['signature_id'].nunique():,} signatures -> {out}"
    )


@app.command("landmarks")
def landmarks(
    gene_info: str = typer.Option(..., help="Path to gene_info table from LINCS"),
    output: str = typer.Option("data/l1000_landmarks.txt", help="Output path for landmark symbols"),
) -> None:
    syms = derive_landmarks_from_gene_info(gene_info)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(syms) + "\n")
    typer.echo(f"Wrote {len(syms)} landmark genes -> {out}")


@app.command("make-target")
def make_target(
    h5ad: Optional[str] = typer.Option(None, help="Optional .h5ad to compute from cluster"),
    cluster_key: Optional[str] = typer.Option(None, help="Obs column for clusters"),
    cluster: Optional[str] = typer.Option(None, help="Cluster label to target"),
    reference: str = typer.Option("rest", help="Reference label or 'rest'"),
    method: str = typer.Option("rank_biserial", help="Method for differential signal"),
    up_file: Optional[str] = typer.Option(None, help="Text file of up genes"),
    down_file: Optional[str] = typer.Option(None, help="Text file of down genes"),
    output: str = typer.Option("target.json", help="Output JSON path for TargetSignature"),
    pseudobulk_key: Optional[str] = typer.Option(
        None,
        help="Optional obs column to aggregate cells into pseudobulk replicates",
    ),
    qc_report: Optional[str] = typer.Option(
        None,
        help="Optional path to write QC summary JSON",
    ),
    library_genes: Optional[str] = typer.Option(
        None,
        help="Optional newline-delimited gene list to estimate library overlap",
    ),
) -> None:
    if h5ad:
        if not (cluster_key and cluster):
            raise typer.BadParameter("--cluster-key and --cluster required with --h5ad")
        adata = load_h5ad(h5ad)
        ts = target_from_cluster(
            adata,
            cluster_key=cluster_key,
            cluster=str(cluster),
            reference=reference,
            method=method,
            pseudobulk_key=pseudobulk_key,
        )
    else:
        if not (up_file or down_file):
            raise typer.BadParameter("Provide --h5ad or at least one of --up-file/--down-file")
        up_genes = (
            [line.strip() for line in Path(up_file).read_text().splitlines() if line.strip()]
            if up_file
            else []
        )
        down_genes = (
            [line.strip() for line in Path(down_file).read_text().splitlines() if line.strip()]
            if down_file
            else []
        )
        ts = target_from_gene_lists(up_genes, down_genes)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    lib_genes = None
    if library_genes and Path(library_genes).exists():
        lib_genes = [
            line.strip()
            for line in Path(library_genes).read_text().splitlines()
            if line.strip()
        ]
    qc_summary = summarize_target_signature(ts, library_genes=lib_genes)
    payload = ts.model_copy(
        update={"metadata": {**ts.metadata, "qc_summary": qc_summary}}
    )
    out.write_text(json.dumps(payload.model_dump()))
    if qc_report:
        Path(qc_report).parent.mkdir(parents=True, exist_ok=True)
        Path(qc_report).write_text(json.dumps(qc_summary, indent=2))
    typer.echo(f"Wrote target to {out}")


@app.command("score")
def score(
    target_json: str = typer.Option(..., help="Path to TargetSignature JSON"),
    library: str = typer.Option(..., help="LINCS long file (csv/tsv/parquet)"),
    method: str = typer.Option("baseline", help="baseline|metric"),
    model_path: Optional[str] = typer.Option(None, help="Checkpoint for metric method"),
    top_k: int = typer.Option(50, help="Top-k rows to return"),
    blend: float = typer.Option(0.5, help="Blend weight for metric"),
    auto_blend: bool = typer.Option(
        False, help="Automatically tune blend between baseline and metric"
    ),
    output: Optional[str] = typer.Option(None, help="Optional output Parquet path"),
    json_output: Optional[str] = typer.Option(
        None, help="Optional output JSON path (includes metadata)"
    ),
    cell_line: Optional[str] = typer.Option(
        None,
        help="Optional cell line filter to reduce library size",
    ),
    cell_lines: Optional[List[str]] = typer.Option(
        None,
        help="Optional list of cell lines (repeat flag)",
        rich_help_panel="Filtering",
    ),
    moa: Optional[str] = typer.Option(None, help="Optional MOA filter"),
    moas: Optional[List[str]] = typer.Option(
        None,
        help="Optional list of MOAs (repeat flag)",
        rich_help_panel="Filtering",
    ),
    pert_type: Optional[str] = typer.Option(
        None,
        help="Optional perturbation type (e.g., TRT_CP)",
    ),
    pert_types: Optional[List[str]] = typer.Option(
        None,
        help="Optional list of perturbation types",
        rich_help_panel="Filtering",
    ),
    compound: Optional[str] = typer.Option(None, help="Optional compound name filter"),
    compounds: Optional[List[str]] = typer.Option(
        None,
        help="Optional list of compounds (repeat flag)",
    ),
    dose_range: Optional[str] = typer.Option(
        None,
        help="Dose range as 'min,max' on pert_dose if available",
        rich_help_panel="Filtering",
    ),
    time_range: Optional[str] = typer.Option(
        None,
        help="Time range as 'min,max' on pert_time if available",
        rich_help_panel="Filtering",
    ),
    touchstone: bool = typer.Option(
        False,
        help="Restrict to Touchstone signatures if 'is_gold' flag available",
    ),
    collapse_replicates: bool = typer.Option(
        False,
        help="Apply MODZ-style replicate collapsing when 'replicate_id' column is present",
        rich_help_panel="Preprocessing",
    ),
) -> None:
    def _summarize(df: pd.DataFrame) -> dict:
        out = {"rows": int(len(df))}
        columns = [
            ("signature_id", "signatures"),
            ("compound", "compounds"),
            ("cell_line", "cell_lines"),
        ]
        for col, key in columns:
            if col in df.columns:
                out[key] = int(df[col].nunique())
        return out

    ts = TargetSignature.model_validate_json(Path(target_json).read_text())
    # If a cell_line(s) filter is provided, attempt predicate pushdown via pyarrow.dataset
    selected_cells: List[str] = []
    if cell_lines:
        selected_cells = [str(x) for x in cell_lines if str(x)]
    if cell_line:
        selected_cells += [str(cell_line)]
    selected_moas: List[str] = []
    if moas:
        selected_moas = [str(x) for x in moas if str(x)]
    if moa:
        selected_moas += [str(moa)]
    selected_ptypes: List[str] = []
    if pert_types:
        selected_ptypes = [str(x) for x in pert_types if str(x)]
    if pert_type:
        selected_ptypes += [str(pert_type)]

    selected_compounds: List[str] = []
    if compounds:
        selected_compounds = [str(x) for x in compounds if str(x)]
    if compound:
        selected_compounds += [str(compound)]

    def _parse_range(r: Optional[str]) -> Optional[tuple[float, float]]:
        if not r:
            return None
        try:
            parts = [p for p in str(r).replace(" ", "").split(",") if p != ""]
            if len(parts) != 2:
                return None
            lo, hi = float(parts[0]), float(parts[1])
            return (min(lo, hi), max(lo, hi))
        except Exception:
            return None

    dose_rng = _parse_range(dose_range)
    time_rng = _parse_range(time_range)

    if any(
        [
            selected_cells,
            selected_moas,
            selected_ptypes,
            selected_compounds,
            dose_rng,
            time_rng,
            touchstone,
        ]
    ):
        try:
            import pyarrow.dataset as ds  # local import; optional

            dataset = ds.dataset(library, format="parquet")
            names = set(dataset.schema.names)
            exprs = []
            warnings: list[str] = []
            if selected_cells and "cell_line" in names:
                exprs.append(ds.field("cell_line").isin(sorted(set(selected_cells))))
            elif selected_cells:
                warnings.append(
                    "Requested cell_line filter but column 'cell_line' not found; ignoring."
                )
            if selected_moas and "moa" in names:
                exprs.append(ds.field("moa").isin(sorted(set(selected_moas))))
            elif selected_moas:
                warnings.append(
                    "Requested MOA filter but column 'moa' not found; ignoring."
                )
            if selected_ptypes and "pert_type" in names:
                exprs.append(ds.field("pert_type").isin(sorted(set(selected_ptypes))))
            elif selected_ptypes:
                warnings.append(
                    "Requested pert_type filter but column 'pert_type' not found; ignoring."
                )
            if selected_compounds and "compound" in names:
                exprs.append(ds.field("compound").isin(sorted(set(selected_compounds))))
            elif selected_compounds:
                warnings.append(
                    "Requested compound filter but column 'compound' not found; ignoring."
                )
            if dose_rng and "pert_dose" in names:
                lo, hi = dose_rng
                exprs.append((ds.field("pert_dose") >= lo) & (ds.field("pert_dose") <= hi))
            elif dose_rng:
                warnings.append(
                    "Requested pert_dose range but column 'pert_dose' not found; ignoring."
                )
            if time_rng and "pert_time" in names:
                lo, hi = time_rng
                exprs.append((ds.field("pert_time") >= lo) & (ds.field("pert_time") <= hi))
            elif time_rng:
                warnings.append(
                    "Requested pert_time range but column 'pert_time' not found; ignoring."
                )
            if touchstone and "is_gold" in names:
                # Accept common truthy forms
                exprs.append(
                    ds.field("is_gold").isin([True, 1, "1", "1.0", "true", "True"])
                )
            elif touchstone:
                warnings.append(
                    "Requested --touchstone but column 'is_gold' not found; ignoring."
                )
            if not exprs:
                # No applicable columns; fall back
                raise RuntimeError("No applicable filter columns present in dataset")
            pre_rows = -1
            try:
                pre_rows = int(dataset.count_rows())
            except Exception:
                pass
            filt = exprs[0]
            for e in exprs[1:]:
                filt = filt & e
            df_long = dataset.scanner(filter=filt).to_table().to_pandas()
            if warnings:
                for w in warnings:
                    typer.echo(f"[warn] {w}", err=True)
            # Print summary of filters and counts
            summary_filters = {
                "cell_line": sorted(set(selected_cells)) if selected_cells else None,
                "moa": sorted(set(selected_moas)) if selected_moas else None,
                "pert_type": sorted(set(selected_ptypes)) if selected_ptypes else None,
                "compound": sorted(set(selected_compounds)) if selected_compounds else None,
                "dose_range": dose_rng,
                "time_range": time_rng,
                "touchstone": bool(touchstone),
            }
            summary_text = ", ".join([f"{k}={v}" for k, v in summary_filters.items() if v])
            if summary_text:
                typer.echo(f"[info] filter summary: {summary_text}", err=True)
            post_stats = _summarize(df_long)
            if pre_rows >= 0:
                typer.echo(
                    f"[info] rows pre-filter={pre_rows:,} post-filter={post_stats['rows']:,}",
                    err=True,
                )
            else:
                typer.echo(
                    f"[info] rows post-filter={post_stats['rows']:,}", err=True
                )
        except Exception:
            # Fallback: load entire table then filter
            df_full = load_lincs_long(library)
            pre_rows = len(df_full)
            df_long = df_full
            if selected_cells and "cell_line" in df_long.columns:
                df_long = df_long[df_long["cell_line"].astype(str).isin(set(selected_cells))]
            elif selected_cells:
                typer.echo(
                    "[warn] Requested cell_line filter but column 'cell_line' not found; ignoring.",
                    err=True,
                )
            if selected_moas and "moa" in df_long.columns:
                df_long = df_long[df_long["moa"].astype(str).isin(set(selected_moas))]
            elif selected_moas:
                typer.echo(
                    "[warn] Requested MOA filter but column 'moa' not found; ignoring.",
                    err=True,
                )
            if selected_ptypes and "pert_type" in df_long.columns:
                df_long = df_long[df_long["pert_type"].astype(str).isin(set(selected_ptypes))]
            elif selected_ptypes:
                typer.echo(
                    "[warn] Requested pert_type filter but column 'pert_type' not found; ignoring.",
                    err=True,
                )
            if selected_compounds and "compound" in df_long.columns:
                df_long = df_long[df_long["compound"].astype(str).isin(set(selected_compounds))]
            elif selected_compounds:
                typer.echo(
                    "[warn] Requested compound filter but column 'compound' not found; ignoring.",
                    err=True,
                )
            if dose_rng and "pert_dose" in df_long.columns:
                lo, hi = dose_rng
                vals = pd.to_numeric(df_long["pert_dose"], errors="coerce")
                df_long = df_long[(vals >= lo) & (vals <= hi)]
            elif dose_rng:
                typer.echo(
                    "[warn] Requested pert_dose range but column 'pert_dose' not found; ignoring.",
                    err=True,
                )
            if time_rng and "pert_time" in df_long.columns:
                lo, hi = time_rng
                vals = pd.to_numeric(df_long["pert_time"], errors="coerce")
                df_long = df_long[(vals >= lo) & (vals <= hi)]
            elif time_rng:
                typer.echo(
                    "[warn] Requested pert_time range but column 'pert_time' not found; ignoring.",
                    err=True,
                )
            if touchstone and "is_gold" in df_long.columns:
                s = df_long["is_gold"].astype(str).str.lower()
                df_long = df_long[s.isin({"1", "true", "yes"}) | (s == "1.0")]
            elif touchstone:
                typer.echo(
                    "[warn] Requested --touchstone but column 'is_gold' not found; ignoring.",
                    err=True,
                )
            df_long = df_long.reset_index(drop=True)
            summary_filters = {
                "cell_line": sorted(set(selected_cells)) if selected_cells else None,
                "moa": sorted(set(selected_moas)) if selected_moas else None,
                "pert_type": sorted(set(selected_ptypes)) if selected_ptypes else None,
                "compound": sorted(set(selected_compounds)) if selected_compounds else None,
                "dose_range": dose_rng,
                "time_range": time_rng,
                "touchstone": bool(touchstone),
            }
            summary_text = ", ".join([f"{k}={v}" for k, v in summary_filters.items() if v])
            if summary_text:
                typer.echo(f"[info] filter summary: {summary_text}", err=True)
            post_stats = _summarize(df_long)
            typer.echo(
                f"[info] rows pre-filter={pre_rows:,} post-filter={post_stats['rows']:,}",
                err=True,
            )
    else:
        df_long = load_lincs_long(library)
    if collapse_replicates and "replicate_id" in df_long.columns:
        df_long = collapse_replicates_modz(df_long)

    res = rank_drugs(
        ts,
        df_long,
        method=method,
        model_path=model_path,
        top_k=top_k,
        blend=blend,
        auto_blend=auto_blend,
    )
    out_df = pd.DataFrame(res.model_dump()["ranking"])  # serialized as list-of-dicts
    # Optional FDR q-values (per query) using Benjamini–Hochberg
    try:
        if not out_df.empty and "p_value" in out_df.columns:
            p = out_df["p_value"].astype(float).to_numpy()
            m = max(1, len(p))
            order = pd.Series(p).sort_values().index.to_numpy()
            ranks = pd.Series(range(1, m + 1), index=order).sort_index().to_numpy()
            q = (p * m / ranks).clip(upper=1.0)
            # enforce monotonicity when sorted by p
            q_sorted = q[order]
            for i in range(m - 2, -1, -1):
                q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
            q_final = pd.Series(index=order, data=q_sorted).sort_index().to_numpy()
            out_df["q_value"] = q_final
    except Exception:
        pass
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(output, engine="pyarrow", index=False)
        typer.echo(f"Wrote results -> {output}")
    else:
        typer.echo(out_df.to_string(index=False))

    if json_output:
        Path(json_output).parent.mkdir(parents=True, exist_ok=True)
        # Compose metadata similar to UI
        blend_val = (
            float(res.metadata.get("blend", blend)) if hasattr(res, "metadata") else float(blend)
        )
        meta = {
            "library": str(library),
            "n": int(len(out_df)),
            "method": method,
            "top_k": int(top_k),
            "auto_blend": bool(auto_blend),
            "blend": blend_val,
            "model_path": model_path,
        }
        payload = {"results": out_df.to_dict(orient="records"), "meta": meta}
        Path(json_output).write_text(json.dumps(payload, indent=2))
        typer.echo(f"Wrote JSON -> {json_output}")


@app.command("train")
def train() -> None:
    subprocess.run(["python", "-m", "scperturb_cmap.models.train"], check=True)


@app.command("evaluate")
def evaluate(checkpoint: str = typer.Option(..., help="Path to checkpoint .pt")) -> None:
    from scperturb_cmap.models.evaluate import evaluate_checkpoint

    metrics = evaluate_checkpoint(checkpoint)
    typer.echo(json.dumps(metrics))


@app.command("ui")
def ui() -> None:
    # Launch the Streamlit UI
    script = Path("src/scperturb_cmap/ui/app.py")
    if not script.exists():
        raise typer.BadParameter(f"UI script not found: {script}")
    # Pass through the LINCS path via --lincs if SCPC_LINCS is set
    args = ["-m", "streamlit", "run", str(script)]
    try:
        subprocess.run(["python", *args], check=True)
    except subprocess.CalledProcessError as e:
        raise typer.Exit(code=e.returncode)


def main() -> None:
    app()
