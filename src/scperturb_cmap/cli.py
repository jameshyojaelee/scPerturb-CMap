from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Optional, List

import pandas as pd
import torch
import typer

from scperturb_cmap import __version__
from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.data.lincs_loader import load_lincs_long
from scperturb_cmap.data.lincs_gctx import gctx_to_long
from scperturb_cmap.data.resources import load_l1000_landmarks, derive_landmarks_from_gene_info
from scperturb_cmap.data.preprocess import harmonize_symbols
from scperturb_cmap.data.scrna_loader import load_h5ad
from scperturb_cmap.data.signatures import target_from_cluster, target_from_gene_lists
from scperturb_cmap.io.schemas import TargetSignature
from scperturb_cmap.io.serde import load_parquet_dataset_filtered
from scperturb_cmap.utils.device import get_device

app = typer.Typer(name="scperturb-cmap", help="scPerturb-CMap command line interface")


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
    input: Optional[str] = typer.Option(None, help="Input LINCS long file (csv/tsv/parquet)"),
    output: str = typer.Option(
        "examples/data/lincs_demo.parquet", help="Output Parquet path"
    ),
    genes_file: Optional[str] = typer.Option(None, help="Optional text file of gene symbols to keep"),
    landmarks: bool = typer.Option(
        False, help="Filter to the L1000 landmark genes"
    ),
    landmarks_file: Optional[str] = typer.Option(None, help="Optional override path to L1000 landmark list"),
    gctx: Optional[str] = typer.Option(None, help="Optional Level 5 GCTX to convert to long format"),
    gene_info: Optional[str] = typer.Option(None, help="Optional gene_info table for mapping IDs to symbols"),
    sig_info: Optional[str] = typer.Option(None, help="Optional sig_info table for metadata (sig_id, cell_id, pert_iname, etc.)"),
    inst_info: Optional[str] = typer.Option(None, help="Optional inst_info table for additional metadata (not required)"),
    repurposing: Optional[str] = typer.Option(None, help="Optional Repurposing Hub annotations (for MOA/targets)"),
    pert_type: Optional[str] = typer.Option(None, help="Optional perturbation type filter (e.g., TRT_CP)"),
    chunk_cols: int = typer.Option(0, help="If >0, write GCTX conversion in column chunks"),
    partition_by: Optional[str] = typer.Option(None, help="Optional column to partition Parquet dataset by (e.g., cell_line)"),
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
                f"Wrote {len(df):,} rows, {df['signature_id'].nunique():,} signatures, {df['gene_symbol'].nunique():,} genes -> {out}"
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
    cell_line: Optional[str] = typer.Option(None, help="Optional cell line filter to reduce library size"),
    cell_lines: Optional[List[str]] = typer.Option(None, help="Optional list of cell lines (repeat flag)", rich_help_panel="Filtering"),
    moa: Optional[str] = typer.Option(None, help="Optional MOA filter"),
    moas: Optional[List[str]] = typer.Option(None, help="Optional list of MOAs (repeat flag)", rich_help_panel="Filtering"),
    pert_type: Optional[str] = typer.Option(None, help="Optional perturbation type (e.g., TRT_CP)"),
    pert_types: Optional[List[str]] = typer.Option(None, help="Optional list of perturbation types", rich_help_panel="Filtering"),
    compound: Optional[str] = typer.Option(None, help="Optional compound name filter"),
    compounds: Optional[List[str]] = typer.Option(None, help="Optional list of compounds (repeat flag)"),
    dose_range: Optional[str] = typer.Option(None, help="Dose range as 'min,max' on pert_dose if available", rich_help_panel="Filtering"),
    time_range: Optional[str] = typer.Option(None, help="Time range as 'min,max' on pert_time if available", rich_help_panel="Filtering"),
    touchstone: bool = typer.Option(False, help="Restrict to Touchstone signatures if 'is_gold' flag available"),
) -> None:
    def _summarize(df: pd.DataFrame) -> dict:
        out = {"rows": int(len(df))}
        for col, key in [("signature_id", "signatures"), ("compound", "compounds"), ("cell_line", "cell_lines")]:
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

    if selected_cells or selected_moas or selected_ptypes or selected_compounds or dose_rng or time_rng or touchstone:
        try:
            import pyarrow.dataset as ds  # local import; optional

            dataset = ds.dataset(library, format="parquet")
            names = set(dataset.schema.names)
            exprs = []
            warnings: list[str] = []
            if selected_cells and "cell_line" in names:
                exprs.append(ds.field("cell_line").isin(sorted(set(selected_cells))))
            elif selected_cells:
                warnings.append("Requested cell_line filter but column 'cell_line' not found; ignoring.")
            if selected_moas and "moa" in names:
                exprs.append(ds.field("moa").isin(sorted(set(selected_moas))))
            elif selected_moas:
                warnings.append("Requested MOA filter but column 'moa' not found; ignoring.")
            if selected_ptypes and "pert_type" in names:
                exprs.append(ds.field("pert_type").isin(sorted(set(selected_ptypes))))
            elif selected_ptypes:
                warnings.append("Requested pert_type filter but column 'pert_type' not found; ignoring.")
            if selected_compounds and "compound" in names:
                exprs.append(ds.field("compound").isin(sorted(set(selected_compounds))))
            elif selected_compounds:
                warnings.append("Requested compound filter but column 'compound' not found; ignoring.")
            if dose_rng and "pert_dose" in names:
                lo, hi = dose_rng
                exprs.append((ds.field("pert_dose") >= lo) & (ds.field("pert_dose") <= hi))
            elif dose_rng:
                warnings.append("Requested pert_dose range but column 'pert_dose' not found; ignoring.")
            if time_rng and "pert_time" in names:
                lo, hi = time_rng
                exprs.append((ds.field("pert_time") >= lo) & (ds.field("pert_time") <= hi))
            elif time_rng:
                warnings.append("Requested pert_time range but column 'pert_time' not found; ignoring.")
            if touchstone and "is_gold" in names:
                # Accept common truthy forms
                exprs.append((ds.field("is_gold") == True) | (ds.field("is_gold") == 1) | (ds.field("is_gold") == "1") | (ds.field("is_gold") == "true") | (ds.field("is_gold") == "True"))
            elif touchstone:
                warnings.append("Requested --touchstone but column 'is_gold' not found; ignoring.")
            if not exprs:
                # No applicable columns; fall back
                raise RuntimeError("No applicable filter columns present in dataset")
            try:
                pre_rows = int(dataset.count_rows())
            except Exception:
                pre_rows = -1
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
            typer.echo(
                f"[info] filter summary: " + \
                ", ".join([f"{k}={v}" for k, v in summary_filters.items() if v]),
                err=True,
            )
            post_stats = _summarize(df_long)
            if pre_rows >= 0:
                typer.echo(f"[info] rows pre-filter={pre_rows:,} post-filter={post_stats['rows']:,}", err=True)
            else:
                typer.echo(f"[info] rows post-filter={post_stats['rows']:,}", err=True)
        except Exception:
            # Fallback: load entire table then filter
            df_full = load_lincs_long(library)
            pre_rows = len(df_full)
            df_long = df_full
            if selected_cells and "cell_line" in df_long.columns:
                df_long = df_long[df_long["cell_line"].astype(str).isin(set(selected_cells))]
            elif selected_cells:
                typer.echo("[warn] Requested cell_line filter but column 'cell_line' not found; ignoring.", err=True)
            if selected_moas and "moa" in df_long.columns:
                df_long = df_long[df_long["moa"].astype(str).isin(set(selected_moas))]
            elif selected_moas:
                typer.echo("[warn] Requested MOA filter but column 'moa' not found; ignoring.", err=True)
            if selected_ptypes and "pert_type" in df_long.columns:
                df_long = df_long[df_long["pert_type"].astype(str).isin(set(selected_ptypes))]
            elif selected_ptypes:
                typer.echo("[warn] Requested pert_type filter but column 'pert_type' not found; ignoring.", err=True)
            if selected_compounds and "compound" in df_long.columns:
                df_long = df_long[df_long["compound"].astype(str).isin(set(selected_compounds))]
            elif selected_compounds:
                typer.echo("[warn] Requested compound filter but column 'compound' not found; ignoring.", err=True)
            if dose_rng and "pert_dose" in df_long.columns:
                lo, hi = dose_rng
                vals = pd.to_numeric(df_long["pert_dose"], errors="coerce")
                df_long = df_long[(vals >= lo) & (vals <= hi)]
            elif dose_rng:
                typer.echo("[warn] Requested pert_dose range but column 'pert_dose' not found; ignoring.", err=True)
            if time_rng and "pert_time" in df_long.columns:
                lo, hi = time_rng
                vals = pd.to_numeric(df_long["pert_time"], errors="coerce")
                df_long = df_long[(vals >= lo) & (vals <= hi)]
            elif time_rng:
                typer.echo("[warn] Requested pert_time range but column 'pert_time' not found; ignoring.", err=True)
            if touchstone and "is_gold" in df_long.columns:
                s = df_long["is_gold"].astype(str).str.lower()
                df_long = df_long[s.isin({"1", "true", "yes"}) | (s == "1.0")]
            elif touchstone:
                typer.echo("[warn] Requested --touchstone but column 'is_gold' not found; ignoring.", err=True)
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
            typer.echo(
                f"[info] filter summary: " + \
                ", ".join([f"{k}={v}" for k, v in summary_filters.items() if v]),
                err=True,
            )
            post_stats = _summarize(df_long)
            typer.echo(f"[info] rows pre-filter={pre_rows:,} post-filter={post_stats['rows']:,}", err=True)
    else:
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
