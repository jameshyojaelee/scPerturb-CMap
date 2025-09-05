from __future__ import annotations

import io
import tempfile
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.data.lincs_loader import load_lincs_long
from scperturb_cmap.data.scrna_loader import load_h5ad
from scperturb_cmap.data.signatures import (
    target_from_cluster,
    target_from_gene_lists,
)
from scperturb_cmap.io.schemas import TargetSignature

st.set_page_config(page_title="scPerturb-CMap Demo", layout="wide")


@st.cache_data(show_spinner=False)
def load_demo_library() -> pd.DataFrame:
    # Try loading example parquet; otherwise synthesize a tiny demo
    demo_path = st.session_state.get("demo_lincs_path", "examples/data/lincs_demo.parquet")
    try:
        return load_lincs_long(demo_path)
    except Exception:
        rng = np.random.default_rng(0)
        genes = [f"G{i}" for i in range(1, 41)]
        rows = []
        for s in range(20):
            for g in genes:
                rows.append(
                    {
                        "signature_id": f"sig{s}",
                        "compound": f"C{s%5}",
                        "cell_line": f"CL{s%3}",
                        "moa": ["classA", "classB"][s % 2],
                        "target": genes[s % len(genes)],
                        "gene_symbol": g,
                        "score": float(rng.normal()),
                    }
                )
        return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def read_uploaded_h5ad(uploaded) -> Optional[object]:
    if uploaded is None:
        return None
    data = uploaded.read()
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        path = tmp.name
    return load_h5ad(path)


def sidebar_controls(
    lincs_long: pd.DataFrame,
) -> Tuple[
    TargetSignature,
    pd.DataFrame,
    str,
    int,
    Optional[str],
    Optional[float],
    Optional[str],
]:
    st.sidebar.header("Data & Target")
    # Target source
    target_mode = st.sidebar.radio(
        "Target source",
        ["Demo", "+ Gene lists", "+ .h5ad"]
    )

    # Build target
    if target_mode == "+ Gene lists":
        up_text = st.sidebar.text_area("Up genes (one per line)", "G1\nG2\nG3")
        down_text = st.sidebar.text_area("Down genes (one per line)", "G10\nG11")
        up_genes = [g.strip() for g in up_text.splitlines() if g.strip()]
        down_genes = [g.strip() for g in down_text.splitlines() if g.strip()]
        target_sig = target_from_gene_lists(up_genes, down_genes)
    elif target_mode == "+ .h5ad":
        h5ad_file = st.sidebar.file_uploader("Upload .h5ad", type=["h5ad"]) 
        adata = read_uploaded_h5ad(h5ad_file)
        if adata is None:
            st.sidebar.info("Upload an .h5ad file to build a target signature.")
            # Default small target
            target_sig = target_from_gene_lists(["G1", "G2"], ["G10"]) 
        else:
            cluster_key = st.sidebar.selectbox("Cluster key", sorted(list(adata.obs.columns)))
            labels = adata.obs[cluster_key].astype(str)
            cluster = st.sidebar.selectbox("Cluster", sorted(labels.unique().tolist()))
            ref_mode = st.sidebar.radio("Reference", ["rest", "cluster"])
            reference = "rest"
            if ref_mode == "cluster":
                ref_label = st.sidebar.selectbox(
                    "Reference cluster", sorted(labels.unique().tolist())
                )
                reference = str(ref_label)
            method = st.sidebar.selectbox("Method", ["rank_biserial", "logfc"]) 
            target_sig = target_from_cluster(
                adata,
                cluster_key=cluster_key,
                cluster=str(cluster),
                reference=reference,
                method=method,
            )
    else:
        # Demo: simple up/down lists overlapping custom demo library
        target_sig = target_from_gene_lists(["G1", "G2", "G3"], ["G10", "G11"]) 

    st.sidebar.header("Scoring")
    method = st.sidebar.selectbox("Method", ["baseline", "metric"]) 
    top_k = int(st.sidebar.slider("Top K", min_value=10, max_value=200, value=50, step=10))
    blend = float(st.sidebar.slider("Blend (metric)", 0.0, 1.0, 0.5))
    model_file = None
    if method == "metric":
        model_upload = st.sidebar.file_uploader("Checkpoint (.pt)", type=["pt"]) 
        if model_upload is not None:
            buf = io.BytesIO(model_upload.read())
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                tmp.write(buf.getvalue())
                tmp.flush()
                model_file = tmp.name
        else:
            st.sidebar.warning("Upload a checkpoint to use metric method.")

    st.sidebar.header("Filter")
    cln = None
    if "cell_line" in lincs_long.columns:
        cln = st.sidebar.selectbox(
            "Cell line (optional)",
            ["All"] + sorted(lincs_long["cell_line"].astype(str).unique().tolist()),
        )
        if cln == "All":
            cln = None

    return (
        target_sig,
        lincs_long,
        method,
        top_k,
        model_file,
        (None if method != "metric" else float(blend)),
        cln,
    )


def plot_signature(ts: TargetSignature, max_genes: int = 10):
    df = pd.DataFrame({"gene": ts.genes, "weight": ts.weights})
    df = df.sort_values("weight")
    neg = df.head(max_genes)
    pos = df.tail(max_genes)
    sub = pd.concat([neg, pos])
    fig = px.bar(
        sub,
        x="gene",
        y="weight",
        color=(sub["weight"] > 0),
        title="Target signature preview",
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("scPerturb-CMap: Connectivity Demo")
    lincs_long = load_demo_library()

    target_sig, lincs_long, method, top_k, model_file, blend, cln = sidebar_controls(
        lincs_long
    )

    # Optional filter by cell line
    # Apply filter if user selected in sidebar

    col1, col2 = st.columns([1, 2])
    with col1:
        plot_signature(target_sig)

    # Filter library by selected cell line if any
    if cln and "cell_line" in lincs_long.columns:
        lib_df = (
            lincs_long[lincs_long["cell_line"].astype(str) == str(cln)].reset_index(
                drop=True
            )
        )
    else:
        lib_df = lincs_long

    with col2:
        try:
            res = rank_drugs(
                target_sig,
                lib_df,
                method=method,
                model_path=model_file,
                top_k=top_k,
                blend=(0.5 if blend is None else float(blend)),
            )
            ranking_df = (
                res.ranking
                if isinstance(res.ranking, pd.DataFrame)
                else pd.DataFrame(res.ranking)
            )
            st.subheader("Results")
            show_cols = [
                c
                for c in [
                    "signature_id",
                    "compound",
                    "moa",
                    "target",
                    "cell_line",
                    "score",
                ]
                if c in ranking_df.columns
            ]
            st.dataframe(ranking_df[show_cols])
            # Export buttons
            csv = ranking_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="results.csv", mime="text/csv")
            parquet_buf = io.BytesIO()
            ranking_df.to_parquet(parquet_buf, engine="pyarrow", index=False)
            st.download_button(
                "Download Parquet",
                data=parquet_buf.getvalue(),
                file_name="results.parquet",
                mime="application/octet-stream",
            )
        except Exception as e:
            st.error(f"Scoring failed: {e}")

    # MOA enrichment
    if "moa" in lib_df.columns and not ranking_df.empty:
        st.subheader("MOA enrichment among top hits")
        counts = ranking_df["moa"].value_counts().reset_index()
        counts.columns = ["moa", "count"]
        fig2 = px.bar(counts, x="moa", y="count")
        st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
