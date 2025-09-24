from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from scperturb_cmap.analysis.enrichment import moa_enrichment
from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.data.lincs_loader import load_lincs_long
from scperturb_cmap.data.scrna_loader import load_h5ad
from scperturb_cmap.data.signatures import (
    summarize_target_signature,
    target_from_cluster,
    target_from_gene_lists,
)
from scperturb_cmap.io.schemas import TargetSignature
from scperturb_cmap.viz.plots import (
    plot_moa_enrichment_bar,
    plot_moa_enrichment_heatmap,
)

st.set_page_config(page_title="scPerturb-CMap Demo", layout="wide")

UI_PRESETS_PATH = Path("examples/data/ui_presets.json")
BOOKMARK_PARAM = "bookmark"
SESSION_VERSION = 1
DEFAULT_UP_TEXT = "G1\nG2\nG3"
DEFAULT_DOWN_TEXT = "G10\nG11"
MAX_SESSION_RESULTS = 250

EXTERNAL_ID_LINKS: Dict[str, Dict[str, str]] = {
    "chembl_id": {
        "label": "ChEMBL",
        "url_template": "https://www.ebi.ac.uk/chembl/compound_report_card/{id}/",
        "display_regex": r"https://www\\.ebi\\.ac\\.uk/chembl/compound_report_card/(CHEMBL[0-9A-Z]+)/",
        "help": "Open compound in the ChEMBL browser.",
    },
    "drugbank_id": {
        "label": "DrugBank",
        "url_template": "https://go.drugbank.com/drugs/{id}",
        "display_regex": r"https://go\\.drugbank\\.com/drugs/(DB\\d+)",
        "help": "Open compound entry on DrugBank.",
    },
    "pubchem_cid": {
        "label": "PubChem",
        "url_template": "https://pubchem.ncbi.nlm.nih.gov/compound/{id}",
        "display_regex": r"https://pubchem\\.ncbi\\.nlm\\.nih\\.gov/compound/(\\d+)",
        "help": "Open compound entry on PubChem.",
    },
    "chebi_id": {
        "label": "ChEBI",
        "url_template": "https://www.ebi.ac.uk/chebi/searchId.do?chebiId={id}",
        "display_regex": r"https://www\\.ebi\\.ac\\.uk/chebi/searchId\\.do\\?chebiId=(CHEBI:\\d+)",
        "help": "Open metabolite entry on ChEBI.",
    },
}


def ensure_session_defaults() -> None:
    default_path = st.session_state.get("demo_lincs_path", "examples/data/lincs_demo.parquet")
    if "library_path" not in st.session_state:
        st.session_state["library_path"] = default_path
    st.session_state.setdefault("target_mode", "Demo")
    st.session_state.setdefault("up_genes_text", DEFAULT_UP_TEXT)
    st.session_state.setdefault("down_genes_text", DEFAULT_DOWN_TEXT)
    st.session_state.setdefault("method", "baseline")
    st.session_state.setdefault("top_k", 50)
    st.session_state.setdefault("blend", 0.5)
    st.session_state.setdefault("cell_line_filter", "All")
    st.session_state.setdefault("active_preset", None)
    st.session_state.setdefault("session_metadata", {})
    st.session_state.setdefault("_bookmark_consumed", False)


@st.cache_data(show_spinner=False)
def load_ui_presets(path: str | Path = UI_PRESETS_PATH) -> Dict[str, Dict[str, Any]]:
    target_path = Path(path)
    if not target_path.exists():
        return {}
    with target_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Preset JSON must be an object mapping names to gene sets.")
    return payload


def parse_gene_block(text: str) -> List[str]:
    if not text:
        return []
    return [gene.strip() for gene in text.splitlines() if gene.strip()]


def encode_state_token(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8")


def decode_state_token(token: str) -> Dict[str, Any]:
    data = base64.urlsafe_b64decode(token.encode("utf-8"))
    return json.loads(data.decode("utf-8"))


def handle_bookmark_on_load() -> None:
    params = st.experimental_get_query_params()
    tokens = params.get(BOOKMARK_PARAM)
    if not tokens:
        return
    if st.session_state.get("_bookmark_consumed", False):
        return
    token = tokens[0]
    try:
        payload = decode_state_token(token)
        payload["source"] = "bookmark"
        apply_state_payload(payload, include_results=False)
        st.session_state["_bookmark_consumed"] = True
        st.experimental_rerun()
    except Exception as exc:  # pragma: no cover - defensive UI path
        st.session_state["_bookmark_consumed"] = True
        st.sidebar.warning(f"Failed to load bookmark: {exc}")


def apply_state_payload(payload: Dict[str, Any], *, include_results: bool) -> None:
    if not isinstance(payload, dict):
        raise ValueError("State payload must be a dictionary.")
    ensure_session_defaults()

    library_path = payload.get("library_path")
    if library_path:
        st.session_state["library_path"] = str(library_path)

    method = payload.get("method")
    if method in {"baseline", "metric"}:
        st.session_state["method"] = method

    top_k = payload.get("top_k")
    if top_k is not None:
        st.session_state["top_k"] = int(top_k)

    blend = payload.get("blend")
    if blend is not None:
        st.session_state["blend"] = float(blend)

    cell_filter = payload.get("cell_line_filter")
    if cell_filter is None or cell_filter == "All":
        st.session_state["cell_line_filter"] = "All"
    elif isinstance(cell_filter, str):
        st.session_state["cell_line_filter"] = cell_filter

    target_context = payload.get("target_context") or {}
    st.session_state["target_context"] = target_context
    st.session_state["target_mode"] = target_context.get(
        "mode",
        st.session_state.get("target_mode", "Demo"),
    )
    st.session_state["active_preset"] = target_context.get("preset")

    gene_lists = target_context.get("gene_lists") or {}
    up_block = "\n".join(gene_lists.get("up_genes", []))
    down_block = "\n".join(gene_lists.get("down_genes", []))
    if up_block:
        st.session_state["up_genes_text"] = up_block
    if down_block:
        st.session_state["down_genes_text"] = down_block

    if target_context.get("cluster_key"):
        st.session_state["target_cluster_key"] = target_context["cluster_key"]
    if target_context.get("cluster"):
        st.session_state["target_cluster_label"] = target_context["cluster"]
    if target_context.get("reference_mode"):
        st.session_state["target_reference_mode"] = target_context["reference_mode"]
    if target_context.get("reference_cluster"):
        st.session_state["target_reference_cluster"] = target_context["reference_cluster"]
    if target_context.get("differential_method"):
        st.session_state["target_cluster_method"] = target_context["differential_method"]

    target_signature = payload.get("target_signature")
    if target_signature:
        st.session_state["target_signature"] = target_signature

    model_checkpoint = payload.get("model_checkpoint")
    if isinstance(model_checkpoint, dict):
        st.session_state["model_checkpoint_label"] = model_checkpoint.get("label")
        st.session_state["model_checkpoint_path"] = model_checkpoint.get("path")
    elif isinstance(model_checkpoint, str):
        st.session_state["model_checkpoint_label"] = model_checkpoint

    if include_results and payload.get("results") is not None:
        try:
            st.session_state["results_df"] = pd.DataFrame(payload["results"])
        except Exception:
            pass

    st.session_state.setdefault("session_metadata", {})
    st.session_state["session_metadata"]["restored_from"] = payload.get("source", "bookmark")


def prepare_link_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df is None or df.empty:
        return df, {}
    table_df = df.copy()
    column_config: Dict[str, Any] = {}
    for col in list(table_df.columns):
        template_key = col.lower()
        if template_key not in EXTERNAL_ID_LINKS:
            continue
        info = EXTERNAL_ID_LINKS[template_key]
        label = info["label"]
        url_template = info["url_template"]
        display_regex = info.get("display_regex")
        help_text = info.get("help")
        insert_idx = table_df.columns.get_loc(col)
        new_key = label if label not in table_df.columns else f"{label} link"

        def _to_url(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, float) and np.isnan(value):
                return ""
            text = str(value).strip()
            if not text or text.lower() == "nan":
                return ""
            return url_template.format(id=text)

        table_df.insert(insert_idx, new_key, table_df[col].map(_to_url))
        table_df = table_df.drop(columns=[col])
        column_config[new_key] = st.column_config.LinkColumn(
            label,
            help=help_text,
            display_text=display_regex,
        )

    return table_df, column_config


def flatten_metadata(prefix: str, value: Any, collector: List[Tuple[str, Any]]) -> None:
    if isinstance(value, dict):
        for key, val in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_metadata(next_prefix, val, collector)
    else:
        collector.append((prefix, value))


def build_export_metadata(
    target_sig: TargetSignature,
    target_context: Dict[str, Any],
    method: str,
    top_k: int,
    blend: Optional[float],
    cell_line: Optional[str],
    library_path: str,
    model_label: Optional[str],
    model_path: Optional[str],
    scoring_meta: Dict[str, Any],
) -> Dict[str, Any]:
    qc_summary = {}
    if isinstance(target_sig.metadata, dict):
        qc_summary = target_sig.metadata.get("qc_summary", {})
    target_source = target_sig.metadata.get("source", target_context) if isinstance(
        target_sig.metadata, dict
    ) else target_context

    up_count = sum(1 for w in target_sig.weights if w > 0)
    down_count = sum(1 for w in target_sig.weights if w < 0)

    metadata: Dict[str, Any] = {
        "version": SESSION_VERSION,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "library_path": library_path,
        "method": method,
        "top_k": int(top_k),
        "cell_line_filter": cell_line or "All",
        "target": {
            "mode": target_source.get("mode"),
            "preset": target_source.get("preset"),
            "n_genes": len(target_sig.genes),
            "n_up": int(up_count),
            "n_down": int(down_count),
            "summary": qc_summary,
        },
    }

    gene_lists = target_source.get("gene_lists") if isinstance(target_source, dict) else None
    if gene_lists:
        metadata["target"]["gene_lists"] = gene_lists

    if method == "metric" and blend is not None:
        metadata["blend"] = float(blend)

    if model_label or model_path:
        metadata["model_checkpoint"] = {"label": model_label, "path": model_path}

    if scoring_meta:
        metadata["scoring"] = scoring_meta

    return metadata


def metadata_to_header_lines(metadata: Dict[str, Any]) -> List[str]:
    flattened: List[Tuple[str, Any]] = []
    flatten_metadata("metadata", metadata, flattened)
    lines = ["# scPerturb-CMap export"]
    for key, value in flattened:
        if isinstance(value, (dict, list)):
            encoded = json.dumps(value, ensure_ascii=True)
        else:
            encoded = str(value)
        lines.append(f"# {key}: {encoded}")
    return lines


def build_session_payload(
    target_sig: TargetSignature,
    target_context: Dict[str, Any],
    method: str,
    top_k: int,
    blend: Optional[float],
    cell_line: Optional[str],
    library_path: str,
    model_label: Optional[str],
    model_path: Optional[str],
    scoring_meta: Dict[str, Any],
    ranking_df: Optional[pd.DataFrame],
    export_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "version": SESSION_VERSION,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "library_path": library_path,
        "method": method,
        "top_k": int(top_k),
        "blend": (float(blend) if blend is not None else None),
        "cell_line_filter": cell_line,
        "model_checkpoint": (
            {"label": model_label, "path": model_path}
            if (model_label or model_path)
            else None
        ),
        "target_context": target_context,
        "target_signature": target_sig.model_dump(),
        "scoring_metadata": scoring_meta,
        "export_metadata": export_metadata,
    }
    if payload["model_checkpoint"] is None:
        payload.pop("model_checkpoint")

    if ranking_df is not None and not ranking_df.empty:
        session_df = ranking_df.head(MAX_SESSION_RESULTS).copy()
        session_records = session_df.where(pd.notna(session_df), None).to_dict(orient="records")
        payload["results"] = session_records
        payload["results_columns"] = list(session_df.columns)
        payload["result_count"] = int(len(ranking_df))

    return payload


def build_export_files(
    ranking_df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> Tuple[bytes, bytes]:
    df_export = ranking_df.copy()
    # Replace NaNs in non-numeric columns for stable export
    for col in df_export.columns:
        if df_export[col].dtype.kind not in {"i", "u", "f"}:
            df_export[col] = df_export[col].astype(object).where(pd.notna(df_export[col]), "")

    header_lines = metadata_to_header_lines(metadata)
    csv_buffer = io.StringIO()
    for line in header_lines:
        csv_buffer.write(line + "\n")
    df_export.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    json_payload = {
        "metadata": metadata,
        "results": df_export.where(pd.notna(df_export), None).to_dict(orient="records"),
    }
    json_bytes = json.dumps(json_payload, indent=2).encode("utf-8")
    return csv_bytes, json_bytes

# Allow passing a default LINCS path via CLI arg `--lincs <path>` or env `SCPC_LINCS`.
try:
    if "--lincs" in sys.argv:
        idx = sys.argv.index("--lincs")
        if idx + 1 < len(sys.argv):
            st.session_state["demo_lincs_path"] = sys.argv[idx + 1]
    elif os.getenv("SCPC_LINCS"):
        st.session_state["demo_lincs_path"] = os.environ["SCPC_LINCS"]
except Exception:
    # Non-fatal: fall back to default demo path
    pass


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
def load_library_from_path(path: str) -> pd.DataFrame:
    return load_lincs_long(path)


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
    str,
    Dict[str, Any],
    Optional[str],
]:
    st.sidebar.header("Data & Target")
    lincs_path_input = st.sidebar.text_input(
        "LINCS long file (parquet/csv)",
        key="library_path",
    )
    lincs_path = lincs_path_input.strip() if lincs_path_input else ""
    library_df = lincs_long
    if lincs_path:
        target_file = Path(lincs_path)
        if target_file.exists():
            try:
                library_df = load_library_from_path(str(target_file))
            except Exception as exc:  # pragma: no cover - UI feedback
                st.sidebar.error(f"Failed to load {target_file.name}: {exc}")
        else:
            st.sidebar.warning(
                "Specified library path does not exist; falling back to demo dataset."
            )

    target_mode = st.sidebar.radio(
        "Target source",
        ["Demo", "+ Gene lists", "+ .h5ad"],
        key="target_mode",
    )

    target_context: Dict[str, Any] = {"mode": target_mode, "preset": st.session_state.get("active_preset")}

    if target_mode == "+ Gene lists":
        up_text = st.sidebar.text_area("Up genes (one per line)", key="up_genes_text")
        down_text = st.sidebar.text_area("Down genes (one per line)", key="down_genes_text")
        up_genes = parse_gene_block(up_text)
        down_genes = parse_gene_block(down_text)
        target_sig = target_from_gene_lists(up_genes, down_genes)
        target_context["gene_lists"] = {"up_genes": up_genes, "down_genes": down_genes}
    elif target_mode == "+ .h5ad":
        h5ad_file = st.sidebar.file_uploader("Upload .h5ad", type=["h5ad"], key="target_h5ad_file")
        adata = read_uploaded_h5ad(h5ad_file)
        if adata is None:
            st.sidebar.info("Upload an .h5ad file to build a target signature.")
            target_sig = target_from_gene_lists(["G1", "G2"], ["G10"])
            target_context["note"] = "Awaiting h5ad upload"
        else:
            obs_keys = sorted(list(map(str, adata.obs.columns)))
            st.session_state.setdefault("target_cluster_key", obs_keys[0] if obs_keys else "")
            cluster_key = st.sidebar.selectbox(
                "Cluster key",
                obs_keys,
                key="target_cluster_key",
            )
            labels = adata.obs[cluster_key].astype(str)
            cluster_options = sorted(labels.unique().tolist())
            st.session_state.setdefault("target_cluster_label", cluster_options[0] if cluster_options else "")
            cluster = st.sidebar.selectbox(
                "Cluster",
                cluster_options,
                key="target_cluster_label",
            )
            ref_mode = st.sidebar.radio("Reference", ["rest", "cluster"], key="target_reference_mode")
            reference = "rest"
            ref_cluster = None
            if ref_mode == "cluster":
                st.session_state.setdefault(
                    "target_reference_cluster",
                    cluster_options[0] if cluster_options else "",
                )
                ref_cluster = st.sidebar.selectbox(
                    "Reference cluster",
                    cluster_options,
                    key="target_reference_cluster",
                )
                reference = str(ref_cluster)
            st.session_state.setdefault("target_cluster_method", "rank_biserial")
            diff_method = st.sidebar.selectbox(
                "Method",
                ["rank_biserial", "logfc"],
                key="target_cluster_method",
            )
            target_context.update(
                {
                    "cluster_key": cluster_key,
                    "cluster": str(cluster),
                    "reference_mode": ref_mode,
                    "reference_cluster": (str(ref_cluster) if ref_cluster else None),
                    "differential_method": diff_method,
                }
            )
            target_sig = target_from_cluster(
                adata,
                cluster_key=cluster_key,
                cluster=str(cluster),
                reference=reference,
                method=diff_method,
            )
    else:
        default_up = parse_gene_block(DEFAULT_UP_TEXT)
        default_down = parse_gene_block(DEFAULT_DOWN_TEXT)
        target_sig = target_from_gene_lists(default_up, default_down)
        target_context["gene_lists"] = {"up_genes": default_up, "down_genes": default_down}

    lib_genes = (
        library_df["gene_symbol"].astype(str).unique().tolist()
        if "gene_symbol" in library_df.columns
        else None
    )
    qc_summary = summarize_target_signature(target_sig, library_genes=lib_genes)
    target_sig.metadata = {
        **target_sig.metadata,
        "qc_summary": qc_summary,
        "source": target_context,
    }
    st.session_state["target_signature"] = target_sig.model_dump()
    st.session_state["target_context"] = target_context

    st.sidebar.header("Scoring")
    method = st.sidebar.selectbox("Method", ["baseline", "metric"], key="method")
    top_k = int(
        st.sidebar.slider(
            "Top K",
            min_value=10,
            max_value=200,
            value=int(st.session_state.get("top_k", 50)),
            step=10,
            key="top_k",
        )
    )
    blend_value = float(
        st.sidebar.slider(
            "Blend (metric)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("blend", 0.5)),
            step=0.05,
            key="blend",
            disabled=(method != "metric"),
        )
    )
    model_file: Optional[str] = None
    model_label: Optional[str] = None
    if method == "metric":
        model_upload = st.sidebar.file_uploader("Checkpoint (.pt)", type=["pt"], key="model_upload")
        if model_upload is not None:
            buf = io.BytesIO(model_upload.read())
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                tmp.write(buf.getvalue())
                tmp.flush()
                model_file = tmp.name
                model_label = getattr(model_upload, "name", os.path.basename(model_file))
            st.session_state["model_checkpoint_path"] = model_file
            st.session_state["model_checkpoint_label"] = model_label
        else:
            model_file = st.session_state.get("model_checkpoint_path")
            model_label = st.session_state.get("model_checkpoint_label")
            if not model_file:
                st.sidebar.warning("Upload a checkpoint to use the metric method.")
    else:
        st.session_state.pop("model_checkpoint_path", None)
        st.session_state.pop("model_checkpoint_label", None)

    st.sidebar.header("Filter")
    cln: Optional[str] = None
    if "cell_line" in library_df.columns:
        options = sorted(library_df["cell_line"].astype(str).unique().tolist())
        raw_choice = st.sidebar.selectbox(
            "Cell line (optional)",
            ["All"] + options,
            key="cell_line_filter",
        )
        cln = None if raw_choice == "All" else str(raw_choice)

    return (
        target_sig,
        library_df,
        method,
        top_k,
        model_file,
        (None if method != "metric" else float(blend_value)),
        cln,
        lincs_path,
        target_context,
        model_label,
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
    # Accept override via environment to point UI to prepared landmark library on HPC
    if os.getenv("SCPC_LINCS") and os.path.exists(os.environ["SCPC_LINCS"]):
        try:
            lincs_long = load_lincs_long(os.environ["SCPC_LINCS"])
            st.session_state["demo_lincs_path"] = os.environ["SCPC_LINCS"]
        except Exception:
            pass

    target_sig, lincs_long, method, top_k, model_file, blend, cln, lincs_path = sidebar_controls(
        lincs_long
    )

    # Presets
    st.sidebar.markdown("### Presets")
    col_a, col_b, col_c = st.sidebar.columns(3)
    emt_clicked = col_a.button("EMT reversal demo")
    ifng_clicked = col_b.button("IFN-high demo")
    tex_clicked = col_c.button("T-cell exhaustion")

    def _load_demo_sets(path: str = "examples/data/demo_gene_sets.json"):
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {
            "EMT_UP": ["VIM", "FN1", "ZEB1", "SNAI1", "ITGA5"],
            "EMT_DN": ["EPCAM", "KRT8", "KRT18", "OCLN", "CLDN4"],
            "IFNG_UP": ["STAT1", "IRF1", "CXCL10", "HLA-A", "ISG15"],
            "IFNG_DN": [],
            "TEX_UP": ["PDCD1", "LAG3", "HAVCR2", "CTLA4", "TIGIT"],
            "TEX_DN": [],
        }

    def _score_from_gene_lists(up, dn, library, method: str = "baseline", top_k: int = 50):
        # Restrict to genes present in the library to avoid zero-overlap issues
        try:
            lib_df = library if isinstance(library, pd.DataFrame) else load_lincs_long(str(library))
            lib_genes = set(lib_df["gene_symbol"].astype(str).str.upper())
            up = [g for g in up if g.upper() in lib_genes]
            dn = [g for g in dn if g.upper() in lib_genes]
        except Exception:
            pass
        if not up and not dn:
            raise ValueError("No overlap between provided gene lists and library genes.")
        ts = target_from_gene_lists(up, dn)
        res = rank_drugs(target_signature=ts, library=library, method=method, top_k=top_k)
        # Ensure DataFrame
        df_rank = (
            res.ranking if isinstance(res.ranking, pd.DataFrame) else pd.DataFrame(res.ranking)
        )
        return ts, df_rank

    if emt_clicked or ifng_clicked or tex_clicked:
        gs = _load_demo_sets()
        library_obj = (
            load_lincs_long(str(lincs_path)) if os.path.exists(str(lincs_path)) else lincs_long
        )
        if emt_clicked:
            up, dn = gs.get("EMT_UP", []), gs.get("EMT_DN", [])
        elif ifng_clicked:
            up, dn = gs.get("IFNG_UP", []), gs.get("IFNG_DN", [])
        else:
            up, dn = gs.get("TEX_UP", []), gs.get("TEX_DN", [])
        with st.spinner("Scoring preset against drug library..."):
            ts_preset, df_rank = _score_from_gene_lists(
                up,
                dn,
                library_obj,
                method="baseline",
                top_k=100,
            )
            st.session_state.target_sig = {"genes": ts_preset.genes, "weights": ts_preset.weights}
            st.session_state.results_df = df_rank

    # Optional filter by cell line
    # Apply filter if user selected in sidebar

    col1, col2 = st.columns([1, 2])
    with col1:
        plot_signature(target_sig)
        summary = (
            target_sig.metadata.get("qc_summary", {})
            if isinstance(target_sig.metadata, dict)
            else {}
        )
        if summary:
            st.markdown("**Target QC**")
            st.dataframe(pd.DataFrame(summary.items(), columns=["metric", "value"]))

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
            # Default checkpoint for metric if none uploaded and available
            if method == "metric" and not model_file:
                default_ckpt = os.environ.get("SCPC_MODEL", "artifacts/best.pt")
                if os.path.exists(default_ckpt):
                    model_file = default_ckpt
                else:
                    st.warning("Metric method requires a checkpoint. Upload one or set SCPC_MODEL.")
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
            st.session_state["results_df"] = ranking_df
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
                    "z_score",
                    "p_value",
                    "q_value",
                ]
                if c in ranking_df.columns
            ]
            st.dataframe(
                ranking_df[show_cols],
                use_container_width=True,
                column_config={
                    "score": st.column_config.NumberColumn(
                        "score",
                        help="Lower implies stronger predicted reversal",
                    ),
                    "moa": st.column_config.TextColumn(
                        "moa", help="Mechanism of action"
                    ),
                    "target": st.column_config.TextColumn(
                        "target",
                        help="Primary target or target family",
                    ),
                },
            )
            # Export buttons; replace NaNs for JSON safety
            df_export = ranking_df.copy()
            for c in df_export.columns:
                if c not in df_export.select_dtypes(include=["number"]).columns:
                    df_export[c] = df_export[c].astype(object).where(pd.notna(df_export[c]), "")
            csv = df_export.to_csv(index=False).encode("utf-8")
            dl_col1, dl_col2 = st.columns(2)
            dl_col1.download_button(
                "Download CSV",
                data=csv,
                file_name="scperturb_cmap_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
            import datetime as _dt
            results_json = {
                "results": df_export.to_dict(orient="records"),
                "meta": {
                    "library": str(lincs_path),
                    "n": int(len(df_export)),
                    "method": method,
                    "top_k": int(top_k),
                    "cell_line_filter": cln if cln else None,
                    "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
                },
            }
            dl_col2.download_button(
                "Download JSON",
                data=json.dumps(results_json, indent=2).encode("utf-8"),
                file_name="scperturb_cmap_results.json",
                mime="application/json",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Scoring failed: {e}")

    # MOA enrichment
    if st.session_state.get("results_df") is not None:
        df_cached = st.session_state["results_df"]
        if not df_cached.empty:
            e_df = moa_enrichment(df_cached, top_n=50)
            st.plotly_chart(plot_moa_enrichment_bar(e_df), use_container_width=True)
            st.plotly_chart(plot_moa_enrichment_heatmap(df_cached), use_container_width=True)


if __name__ == "__main__":
    main()
