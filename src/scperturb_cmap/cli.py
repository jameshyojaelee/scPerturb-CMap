from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from scperturb_cmap import __version__
from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.data.lincs_loader import load_lincs_long
from scperturb_cmap.data.preprocess import harmonize_symbols
from scperturb_cmap.data.scrna_loader import load_h5ad
from scperturb_cmap.data.signatures import target_from_cluster, target_from_gene_lists
from scperturb_cmap.io.schemas import TargetSignature
from scperturb_cmap.utils.device import get_device

app = typer.Typer(help="scPerturb-CMap command line interface")


@app.command()
def version() -> None:
    """Print package version."""
    typer.echo(__version__)


@app.command()
def device() -> None:
    """Print the selected compute device (cuda|mps|cpu)."""
    typer.echo(get_device())


@app.command("prepare-lincs")
def prepare_lincs(
    input: str = typer.Option(..., help="Input LINCS long file (csv/tsv/parquet)"),
    output: str = typer.Option(
        "examples/data/lincs_demo.parquet", help="Output Parquet path"
    ),
    genes_file: Optional[str] = typer.Option(None, help="Optional file with gene list"),
    landmarks: bool = typer.Option(
        False, help="Filter to a minimal built-in set of landmark-like genes"
    ),
) -> None:
    df = load_lincs_long(input)
    if genes_file or landmarks:
        if genes_file:
            genes = [
                line.strip()
                for line in Path(genes_file).read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]
        else:
            genes = ["GAPDH", "ACTB", "TUBB", "RPLP0", "EEF1A1"]
        genes = harmonize_symbols(genes)
        df = df.assign(
            gsym=df["gene_symbol"].astype(str).str.strip().str.upper()
        )
        df = df[df["gsym"].isin(set(genes))].drop(columns=["gsym"]).reset_index(drop=True)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)
    typer.echo(
        f"Wrote {len(df):,} rows, {df['signature_id'].nunique():,} signatures -> {out}"
    )


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
) -> None:
    if h5ad:
        if not (cluster_key and cluster):
            raise typer.BadParameter("--cluster-key and --cluster required with --h5ad")
        adata = load_h5ad(h5ad)
        ts = target_from_cluster(
            adata, cluster_key=cluster_key, cluster=str(cluster), reference=reference, method=method
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
    out.write_text(json.dumps(ts.model_dump()))
    typer.echo(f"Wrote target to {out}")


@app.command("score")
def score(
    target_json: str = typer.Option(..., help="Path to TargetSignature JSON"),
    library: str = typer.Option(..., help="LINCS long file (csv/tsv/parquet)"),
    method: str = typer.Option("baseline", help="baseline|metric"),
    model_path: Optional[str] = typer.Option(None, help="Checkpoint for metric method"),
    top_k: int = typer.Option(50, help="Top-k rows to return"),
    blend: float = typer.Option(0.5, help="Blend weight for metric"),
    output: Optional[str] = typer.Option(None, help="Optional output Parquet path"),
) -> None:
    ts = TargetSignature.model_validate_json(Path(target_json).read_text())
    df_long = load_lincs_long(library)
    res = rank_drugs(ts, df_long, method=method, model_path=model_path, top_k=top_k, blend=blend)
    out_df = pd.DataFrame(res.model_dump()["ranking"])  # serialized as list-of-dicts
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(output, engine="pyarrow", index=False)
        typer.echo(f"Wrote results -> {output}")
    else:
        typer.echo(out_df.to_string(index=False))


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
    typer.echo("UI placeholder: Streamlit app will be added later.")


def main() -> None:
    app()
