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
from scperturb_cmap.analysis.power import (
    bootstrap_rank_confidence,
    compute_signature_stability,
    estimate_signature_sample_size,
    permutation_significance_test,
    recommend_min_cells_per_cluster,
    simulate_false_discovery_rate,
)
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

st.set_page_config(page_title="scPerturb-CMap Demo", page_icon="🧬", layout="wide")

# Consistent Plotly theming
px.defaults.template = "plotly_dark"
# Set default discrete color sequence for consistency across charts
px.defaults.color_discrete_sequence = [
    "#38bdf8",
    "#6366f1",
    "#f97316",
    "#22d3ee",
    "#a78bfa",
]

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


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --scpc-bg: #090f1f;
            --scpc-surface: #111a2e;
            --scpc-surface-alt: #16213d;
            --scpc-border: #1f2b47;
            --scpc-text: #e2e8f0;
            --scpc-muted: #94a3b8;
            --scpc-primary: linear-gradient(120deg, #38bdf8, #6366f1);
            --scpc-primary-solid: #38bdf8;
            --scpc-accent: #f97316;
        }
        body, .stApp {
            font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
            background-color: var(--scpc-bg);
            color: var(--scpc-text);
        }
        .stApp header, .stApp [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stSidebar"] > div:first-child {
            background: var(--scpc-surface);
            border-right: 1px solid var(--scpc-border);
        }
        [data-testid="stSidebar"] * {
            color: var(--scpc-text) !important;
        }
        .stMarkdown,
        .stText,
        .stTextInput,
        .stNumberInput,
        .stSelectbox,
        .stDataFrame,
        .stDataEditor {
            color: var(--scpc-text) !important;
        }
        .stTextInput > div > div > input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div {
            background: var(--scpc-surface-alt);
            color: var(--scpc-text);
            border-radius: 6px;
            border: 1px solid var(--scpc-border);
        }
        .stSlider > div[data-baseweb="slider"] > div {
            background: var(--scpc-border);
        }
        .stSlider [role="slider"] {
            background: var(--scpc-primary-solid);
        }
        .stButton button {
            border-radius: 6px;
            border: none;
            color: #ffffff !important;
            background-image: var(--scpc-primary);
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(56, 189, 248, 0.45);
        }
        .stDataFrame thead tr th, .stDataEditor thead tr th {
            background: var(--scpc-surface-alt) !important;
            color: var(--scpc-text) !important;
            font-weight: 600 !important;
            border-bottom: 1px solid var(--scpc-border) !important;
        }
        .stDataFrame tbody tr, .stDataEditor tbody tr {
            background: var(--scpc-surface) !important;
        }
        .stDataFrame tbody tr:nth-child(even), .stDataEditor tbody tr:nth-child(even) {
            background: var(--scpc-surface-alt) !important;
        }
        .main > div {
            background: transparent;
        }
        .stMetric {
            background: var(--scpc-surface-alt);
            border-radius: 10px;
            border: 1px solid var(--scpc-border);
            box-shadow: 0 15px 35px rgba(13, 20, 40, 0.35);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: var(--scpc-muted);
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--scpc-text);
        }
        .stTabs [data-baseweb="tab"] [role="tab"] {
            border-bottom: 2px solid transparent;
        }
        .stTabs [aria-selected="true"] [role="tab"] {
            border-bottom: 2px solid var(--scpc-primary-solid);
            color: var(--scpc-text);
        }
        code, pre {
            background: rgba(15, 23, 42, 0.75) !important;
            color: #e0f2fe !important;
        }
        .stPlotlyChart {
            background: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def parse_int_list(text: str) -> List[int]:
    if not text:
        return []
    parts = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]
    out: List[int] = []
    for part in parts:
        try:
            out.append(int(part))
        except ValueError:
            raise ValueError(f"Could not parse integer from '{part}'")
    return out


def parse_float_list(text: str) -> List[float]:
    if not text:
        return []
    parts = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]
    values: List[float] = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            raise ValueError(f"Could not parse float from '{part}'")
    return values


def encode_state_token(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8")


def decode_state_token(token: str) -> Dict[str, Any]:
    data = base64.urlsafe_b64decode(token.encode("utf-8"))
    return json.loads(data.decode("utf-8"))


def handle_bookmark_on_load() -> None:
    if st.session_state.get("_bookmark_consumed", False):
        return

    params = st.query_params
    tokens = params.get(BOOKMARK_PARAM)
    if not tokens:
        return
    if isinstance(tokens, (list, tuple)):
        token = tokens[0] if tokens else ""
    else:
        token = tokens
    if not token:
        return
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


def apply_preset_signature(
    name: str,
    preset_payload: Dict[str, Any],
    library_df: pd.DataFrame,
    top_k: int,
) -> Tuple[TargetSignature, pd.DataFrame]:
    up_genes = list(map(str, preset_payload.get("up_genes", [])))
    down_genes = list(map(str, preset_payload.get("down_genes", [])))
    if not up_genes and not down_genes:
        raise ValueError(f"Preset '{name}' is missing gene lists.")

    lib_genes: Optional[set[str]] = None
    if "gene_symbol" in library_df.columns:
        lib_genes = set(library_df["gene_symbol"].astype(str).str.upper())

    def _restrict(genes: List[str]) -> List[str]:
        if lib_genes is None:
            return genes
        overlapping = [g for g in genes if g.upper() in lib_genes]
        return overlapping if overlapping else genes

    restricted_up = _restrict(up_genes)
    restricted_down = _restrict(down_genes)

    target_sig = target_from_gene_lists(restricted_up, restricted_down)
    qc_summary = summarize_target_signature(
        target_sig,
        library_genes=(list(lib_genes) if lib_genes is not None else None),
    )
    source_context = {
        "mode": "Preset",
        "preset": name,
        "gene_lists": {"up_genes": restricted_up, "down_genes": restricted_down},
    }
    target_sig.metadata = {
        **target_sig.metadata,
        "qc_summary": qc_summary,
        "source": source_context,
    }

    res = rank_drugs(target_sig, library_df, method="baseline", top_k=int(top_k))
    ranking_df = res.ranking if isinstance(res.ranking, pd.DataFrame) else pd.DataFrame(res.ranking)

    st.session_state["target_mode"] = "+ Gene lists"
    st.session_state["up_genes_text"] = "\n".join(restricted_up)
    st.session_state["down_genes_text"] = "\n".join(restricted_down)
    st.session_state["active_preset"] = name
    st.session_state["method"] = "baseline"
    st.session_state["blend"] = 0.5
    st.session_state["target_signature"] = target_sig.model_dump()
    st.session_state["target_context"] = source_context
    st.session_state["results_df"] = ranking_df
    st.session_state.setdefault("session_metadata", {})["last_preset"] = name

    return target_sig, ranking_df

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


@st.cache_data(show_spinner=False)
def read_uploaded_table(uploaded) -> Optional[pd.DataFrame]:
    if uploaded is None:
        return None
    uploaded.seek(0)
    name = uploaded.name.lower()
    buffer = io.BytesIO(uploaded.read())
    buffer.seek(0)
    try:
        if name.endswith('.parquet') or name.endswith('.pq'):
            return pd.read_parquet(buffer)
        if name.endswith('.tsv') or name.endswith('.txt'):
            return pd.read_csv(buffer, sep='\t')
        if name.endswith('.json') or name.endswith('.jsonl'):
            buffer.seek(0)
            return pd.read_json(buffer)
        buffer.seek(0)
        return pd.read_csv(buffer)
    except Exception as exc:
        raise ValueError(f"Failed to load uploaded table '{uploaded.name}': {exc}") from exc


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

    target_context: Dict[str, Any] = {
        "mode": target_mode,
        "preset": st.session_state.get("active_preset"),
    }

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
            st.session_state.pop("uploaded_adata", None)
        else:
            st.session_state["uploaded_adata"] = adata
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
            st.session_state.setdefault(
                "target_cluster_label",
                cluster_options[0] if cluster_options else "",
            )
            cluster = st.sidebar.selectbox(
                "Cluster",
                cluster_options,
                key="target_cluster_label",
            )
            ref_mode = st.sidebar.radio(
                "Reference",
                ["rest", "cluster"],
                key="target_reference_mode",
            )
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
            step=10,
            key="top_k",
        )
    )
    blend_value = float(
        st.sidebar.slider(
            "Blend (metric)",
            min_value=0.0,
            max_value=1.0,
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
    sub["direction"] = np.where(sub["weight"] >= 0, "Up", "Down")
    fig = px.bar(
        sub,
        x="gene",
        y="weight",
        color="direction",
        color_discrete_map={"Up": "#38bdf8", "Down": "#f97316"},
        title="Target signature preview",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        xaxis_title="Gene",
        yaxis_title="Weight",
    )
    st.plotly_chart(fig, width="stretch")


def main():
    ensure_session_defaults()
    handle_bookmark_on_load()
    apply_theme()

    st.title("scPerturb-CMap: Connectivity Demo")

    base_library = load_demo_library()
    env_lincs = os.getenv("SCPC_LINCS")
    if env_lincs and os.path.exists(env_lincs):
        try:
            base_library = load_lincs_long(env_lincs)
            st.session_state["demo_lincs_path"] = env_lincs
        except Exception:  # pragma: no cover - defensive fallback path
            st.sidebar.warning(
                "Unable to load dataset referenced by SCPC_LINCS; falling back to demo library."
            )

    (
        target_sig,
        library_df,
        method,
        top_k,
        model_file,
        blend,
        cln,
        lincs_path,
        target_context,
        model_label,
    ) = sidebar_controls(base_library)

    stored_sig_payload = st.session_state.get("target_signature")
    if stored_sig_payload:
        try:
            target_sig = TargetSignature(**stored_sig_payload)
        except Exception:
            pass

    if isinstance(target_sig.metadata, dict) and target_sig.metadata.get("source"):
        target_context = target_sig.metadata["source"]

    presets = load_ui_presets()
    if presets:
        st.sidebar.markdown("### Curated presets")
        preset_columns = st.sidebar.columns(len(presets))
        for (preset_name, preset_payload), column in zip(presets.items(), preset_columns):
            if column.button(preset_name, width="stretch"):
                try:
                    target_sig, preset_df = apply_preset_signature(
                        preset_name,
                        preset_payload,
                        library_df,
                        top_k=top_k,
                    )
                    target_context = target_sig.metadata.get("source", target_context)
                    method = "baseline"
                    blend = None
                    model_file = None
                    model_label = None
                    st.session_state["results_df"] = preset_df
                    st.toast(f"Preset '{preset_name}' applied", icon="✨")
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.sidebar.error(f"Failed to run preset '{preset_name}': {exc}")

    filtered_library = library_df
    if cln and "cell_line" in library_df.columns:
        mask = library_df["cell_line"].astype(str) == str(cln)
        filtered_library = library_df.loc[mask].reset_index(drop=True)

    selected_library_path = (
        lincs_path
        or st.session_state.get("library_path")
        or st.session_state.get("demo_lincs_path")
        or "examples/data/lincs_demo.parquet"
    )

    ranking_df: Optional[pd.DataFrame] = None
    score_metadata: Dict[str, Any] = {}
    scoring_error: Optional[str] = None

    blend_arg = float(blend) if blend is not None else 0.5

    if method == "metric" and not model_file:
        default_ckpt = os.environ.get("SCPC_MODEL", "workspace/artifacts/best.pt")
        if os.path.exists(default_ckpt):
            model_file = default_ckpt
            model_label = os.path.basename(default_ckpt)
        else:
            scoring_error = "Metric method requires a checkpoint. Upload one or set SCPC_MODEL."

    if scoring_error is None:
        try:
            res = rank_drugs(
                target_sig,
                filtered_library,
                method=method,
                model_path=model_file,
                top_k=top_k,
                blend=blend_arg,
            )
            ranking_df = (
                res.ranking
                if isinstance(res.ranking, pd.DataFrame)
                else pd.DataFrame(res.ranking)
            )
            score_metadata = res.metadata or {}
            st.session_state["results_df"] = ranking_df
            st.session_state["last_scoring_meta"] = score_metadata
            st.session_state.setdefault("session_metadata", {})["last_scored_at"] = (
                datetime.utcnow().isoformat(timespec="seconds") + "Z"
            )
        except Exception as exc:
            scoring_error = str(exc)

    if ranking_df is None:
        cached = st.session_state.get("results_df")
        if isinstance(cached, pd.DataFrame):
            ranking_df = cached
        elif cached is not None:
            ranking_df = pd.DataFrame(cached)

    export_metadata: Dict[str, Any] = {}
    csv_bytes: Optional[bytes] = None
    json_bytes: Optional[bytes] = None
    session_payload: Optional[Dict[str, Any]] = None
    session_bytes: Optional[bytes] = None

    if scoring_error:
        message = (
            f"Scoring issue: {scoring_error}. Showing last available results."
            if ranking_df is not None and not ranking_df.empty
            else f"Scoring failed: {scoring_error}"
        )
        st.warning(message)

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
            st.dataframe(
                pd.DataFrame(summary.items(), columns=["metric", "value"]),
                width="stretch",
            )
        info_rows: List[Tuple[str, Any]] = []
        if isinstance(target_context, dict):
            if target_context.get("preset"):
                info_rows.append(("Preset", target_context["preset"]))
            mode_hint = target_context.get("mode", st.session_state.get("target_mode"))
            info_rows.append(("Mode", mode_hint))
        info_rows.append(("Up genes", sum(1 for w in target_sig.weights if w > 0)))
        info_rows.append(("Down genes", sum(1 for w in target_sig.weights if w < 0)))
        info_rows.append(("Library", Path(str(selected_library_path)).name))
        info_df = pd.DataFrame(info_rows, columns=["attribute", "value"])
        info_df["value"] = info_df["value"].apply(lambda x: "" if x is None else str(x))
        st.dataframe(info_df, hide_index=True, width="stretch")

    with col2:
        if ranking_df is None or ranking_df.empty:
            st.info(
                "No results available yet. Adjust the target or scoring parameters to recompute."
            )
        else:
            base_columns = [
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
            external_cols = [
                c for c in ranking_df.columns if c.lower() in EXTERNAL_ID_LINKS
            ]
            other_cols = [
                c
                for c in ranking_df.columns
                if c not in base_columns and c not in external_cols
            ]
            ordered_cols = [
                c for c in base_columns if c in ranking_df.columns
            ] + external_cols + other_cols
            table_df = ranking_df[ordered_cols]
            table_df, link_config = prepare_link_columns(table_df)
            column_config: Dict[str, Any] = {
                "score": st.column_config.NumberColumn(
                    "score",
                    help="Lower implies stronger predicted reversal",
                    format="%.3f",
                ),
                "z_score": st.column_config.NumberColumn("z_score", format="%.2f"),
                "p_value": st.column_config.NumberColumn("p_value", format="%.2e"),
                "q_value": st.column_config.NumberColumn("q_value", format="%.2e"),
                "moa": st.column_config.TextColumn("moa", help="Mechanism of action"),
                "target": st.column_config.TextColumn(
                    "target", help="Primary target or target family"
                ),
            }
            column_config.update(link_config)
            st.data_editor(
                table_df,
                hide_index=True,
                width="stretch",
                column_config=column_config,
                disabled=True,
            )

            export_metadata = build_export_metadata(
                target_sig,
                target_context if isinstance(target_context, dict) else {},
                method,
                top_k,
                (blend if method == "metric" else None),
                cln,
                str(selected_library_path),
                model_label,
                model_file,
                score_metadata,
            )
            csv_bytes, json_bytes = build_export_files(ranking_df, export_metadata)
            session_payload = build_session_payload(
                target_sig,
                target_context if isinstance(target_context, dict) else {},
                method,
                top_k,
                (blend if method == "metric" else None),
                cln,
                str(selected_library_path),
                model_label,
                model_file,
                score_metadata,
                ranking_df,
                export_metadata,
            )
            session_bytes = json.dumps(session_payload, indent=2).encode("utf-8")
            st.session_state["session_snapshot"] = session_payload

            dl_col1, dl_col2 = st.columns(2)
            dl_col1.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="scperturb_cmap_results.csv",
                mime="text/csv",
                width="stretch",
            )
            dl_col2.download_button(
                "Download JSON",
                data=json_bytes,
                file_name="scperturb_cmap_results.json",
                mime="application/json",
                width="stretch",
            )

    st.markdown("### Power & QC Analysis")
    power_tabs = st.tabs(
        [
            "Sample Size",
            "Min Cells",
            "Ranking CIs",
            "Stability",
            "FDR Simulation",
            "Permutation Test",
        ]
    )

    uploaded_adata = st.session_state.get("uploaded_adata")
    cluster_key = st.session_state.get("target_cluster_key")
    cluster_label = st.session_state.get("target_cluster_label")
    ref_mode = st.session_state.get("target_reference_mode", "rest")
    ref_cluster = st.session_state.get("target_reference_cluster")
    reference_value = ref_cluster if ref_mode == "cluster" and ref_cluster else "rest"

    with power_tabs[0]:
        st.write("Estimate how many cells are needed for a stable target signature.")
        if uploaded_adata is None:
            st.info("Upload an .h5ad file in the sidebar to enable sample size estimation.")
        elif not cluster_key or not cluster_label:
            st.info("Select a cluster in the sidebar to evaluate sample sizes.")
        else:
            defaults = st.session_state.get("power_sample_form", {})
            with st.form("power_sample_size_form"):
                sample_sizes_text = st.text_input(
                    "Sample sizes (comma separated, optional)",
                    value=defaults.get("sample_sizes_text", ""),
                )
                replicates_val = st.number_input(
                    "Bootstrap replicates",
                    min_value=1,
                    max_value=500,
                    value=int(defaults.get("replicates", 30)),
                    step=1,
                )
                threshold_val = st.slider(
                    "Correlation threshold",
                    min_value=0.1,
                    max_value=0.99,
                    value=float(defaults.get("threshold", 0.7)),
                    step=0.05,
                )
                corr_options = ["spearman", "pearson", "cosine"]
                selected_corr = defaults.get("correlation_metric", "spearman")
                if selected_corr not in corr_options:
                    selected_corr = "spearman"
                corr_metric = st.selectbox(
                    "Correlation metric",
                    corr_options,
                    index=corr_options.index(selected_corr),
                )

                diff_options = ["rank_biserial", "logfc"]
                default_method = defaults.get(
                    "method",
                    st.session_state.get("target_cluster_method", "rank_biserial"),
                )
                if default_method not in diff_options:
                    default_method = "rank_biserial"
                diff_method = st.selectbox(
                    "Differential method",
                    diff_options,
                    index=diff_options.index(default_method),
                )

                target_context = st.session_state.get("target_context", {})
                pseudobulk_default = target_context.get("pseudobulk_key")
                pseudobulk_val = st.text_input(
                    "Pseudobulk key (optional)",
                    value=defaults.get("pseudobulk_key", pseudobulk_default or ""),
                )
                submitted = st.form_submit_button("Estimate sample size")

            if submitted:
                try:
                    sizes = parse_int_list(sample_sizes_text)
                except ValueError as exc:
                    st.error(f"Sample sizes invalid: {exc}")
                else:
                    with st.spinner("Bootstrapping signatures..."):
                        try:
                            result = estimate_signature_sample_size(
                                uploaded_adata,
                                cluster_key=cluster_key,
                                cluster=str(cluster_label),
                                reference=reference_value,
                                sample_sizes=sizes or None,
                                replicates=int(replicates_val),
                                method=diff_method,
                                pseudobulk_key=pseudobulk_val or None,
                                correlation_metric=corr_metric,
                                threshold=float(threshold_val),
                            )
                        except Exception as exc:
                            st.error(f"Failed to estimate sample size: {exc}")
                        else:
                            st.session_state["power_sample_size_result"] = {
                                "summary": result.summary,
                                "history": result.history,
                                "recommended": result.recommended_size,
                                "threshold": result.threshold,
                                "baseline": result.baseline_cells,
                            }
                            st.session_state["power_sample_form"] = {
                                "sample_sizes_text": sample_sizes_text,
                                "replicates": int(replicates_val),
                                "threshold": float(threshold_val),
                                "correlation_metric": corr_metric,
                                "method": diff_method,
                                "pseudobulk_key": pseudobulk_val,
                            }

            saved = st.session_state.get("power_sample_size_result")
            if saved:
                st.metric(
                    "Recommended target cells",
                    f"{int(saved['recommended'])}",
                    help=f"Baseline cells: {saved['baseline']}; threshold: {saved['threshold']}"
                )
                st.dataframe(saved["summary"], hide_index=True, use_container_width=True)
                if not saved["summary"].empty:
                    fig = px.line(
                        saved["summary"],
                        x="sample_size",
                        y="median_correlation",
                        markers=True,
                        title="Median correlation vs. sample size",
                    )
                    fig.add_hline(
                        y=saved["threshold"],
                        line_dash="dash",
                        annotation_text="Threshold",
                        annotation_position="bottom right",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with st.expander("Bootstrap history", expanded=False):
                    st.dataframe(saved["history"], hide_index=True, use_container_width=True)

    with power_tabs[1]:
        st.write("Summarise minimum recommended cells per cluster for the dataset.")
        if uploaded_adata is None:
            st.info("Upload an .h5ad file in the sidebar to compute per-cluster recommendations.")
        elif not cluster_key:
            st.info("Select a cluster key in the sidebar to evaluate clusters.")
        else:
            defaults = st.session_state.get("power_min_cells_form", {})
            with st.form("power_min_cells_form"):
                sample_sizes_text = st.text_input(
                    "Sample sizes (comma separated, optional)",
                    value=defaults.get("sample_sizes_text", ""),
                )
                replicates_val = st.number_input(
                    "Bootstrap replicates",
                    min_value=1,
                    max_value=500,
                    value=int(defaults.get("replicates", 20)),
                    step=1,
                )
                threshold_val = st.slider(
                    "Correlation threshold",
                    min_value=0.1,
                    max_value=0.99,
                    value=float(defaults.get("threshold", 0.7)),
                    step=0.05,
                )
                corr_options = ["spearman", "pearson", "cosine"]
                selected_corr = defaults.get("correlation_metric", "spearman")
                if selected_corr not in corr_options:
                    selected_corr = "spearman"
                corr_metric = st.selectbox(
                    "Correlation metric",
                    corr_options,
                    index=corr_options.index(selected_corr),
                )

                diff_options = ["rank_biserial", "logfc"]
                default_method = defaults.get(
                    "method",
                    st.session_state.get("target_cluster_method", "rank_biserial"),
                )
                if default_method not in diff_options:
                    default_method = "rank_biserial"
                diff_method = st.selectbox(
                    "Differential method",
                    diff_options,
                    index=diff_options.index(default_method),
                )

                target_context = st.session_state.get("target_context", {})
                pseudobulk_default = target_context.get("pseudobulk_key")
                pseudobulk_val = st.text_input(
                    "Pseudobulk key (optional)",
                    value=defaults.get("pseudobulk_key", pseudobulk_default or ""),
                )
                submitted = st.form_submit_button("Recommend minimum cells")

            if submitted:
                try:
                    sizes = parse_int_list(sample_sizes_text)
                except ValueError as exc:
                    st.error(f"Sample sizes invalid: {exc}")
                else:
                    with st.spinner("Evaluating clusters..."):
                        try:
                            result = recommend_min_cells_per_cluster(
                                uploaded_adata,
                                cluster_key=cluster_key,
                                reference=reference_value,
                                sample_sizes=sizes or None,
                                replicates=int(replicates_val),
                                threshold=float(threshold_val),
                                method=diff_method,
                                pseudobulk_key=pseudobulk_val or None,
                                correlation_metric=corr_metric,
                            )
                        except Exception as exc:
                            st.error(f"Failed to recommend cells: {exc}")
                        else:
                            st.session_state["power_min_cells_result"] = result
                            st.session_state["power_min_cells_form"] = {
                                "sample_sizes_text": sample_sizes_text,
                                "replicates": int(replicates_val),
                                "threshold": float(threshold_val),
                                "correlation_metric": corr_metric,
                                "method": diff_method,
                                "pseudobulk_key": pseudobulk_val,
                            }

            saved = st.session_state.get("power_min_cells_result")
            if saved and isinstance(saved, dict):
                rec_df = saved.get("recommendations")
                if isinstance(rec_df, pd.DataFrame) and not rec_df.empty:
                    st.dataframe(rec_df, hide_index=True, use_container_width=True)
                    valid_clusters = [
                        c
                        for c in rec_df["cluster"].tolist()
                        if c in saved.get("summaries", {})
                    ]
                    if valid_clusters:
                        selected_cluster = st.selectbox(
                            "Inspect cluster details",
                            valid_clusters,
                            key="power_min_cells_cluster_select",
                        )
                        st.markdown("**Summary**")
                        st.dataframe(
                            saved["summaries"][selected_cluster],
                            hide_index=True,
                            use_container_width=True,
                        )
                        with st.expander("Bootstrap history", expanded=False):
                            st.dataframe(
                                saved["histories"][selected_cluster],
                                hide_index=True,
                                use_container_width=True,
                            )
                else:
                    st.info("No recommendations computed yet.")

    with power_tabs[2]:
        st.write("Bootstrap confidence intervals for rankings using replicate-level scores.")
        upload = st.file_uploader(
            "Replicate-level scores (parquet/csv/tsv)",
            type=["csv", "tsv", "txt", "parquet", "pq", "json", "jsonl"],
            key="power_rank_ci_upload",
        )
        rank_df: Optional[pd.DataFrame] = None
        if upload is not None:
            try:
                rank_df = read_uploaded_table(upload)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state["power_rank_ci_table"] = rank_df
        elif "power_rank_ci_table" in st.session_state:
            rank_df = st.session_state.get("power_rank_ci_table")

        if isinstance(rank_df, pd.DataFrame):
            st.dataframe(rank_df.head(20), hide_index=True, use_container_width=True)
            defaults = st.session_state.get("power_rank_ci_form", {})
            with st.form("power_rank_ci_form"):
                id_col = st.text_input(
                    "Signature column",
                    value=defaults.get("id_col", "signature_id"),
                )
                score_col = st.text_input(
                    "Score column",
                    value=defaults.get("score_col", "score"),
                )
                replicate_col = st.text_input(
                    "Replicate column",
                    value=defaults.get("replicate_col", "replicate_id"),
                )
                aggfunc = st.selectbox(
                    "Aggregate replicates with",
                    ["mean", "median", "sum"],
                    index=["mean", "median", "sum"].index(
                        defaults.get("aggfunc", "mean")
                        if defaults.get("aggfunc", "mean") in {"mean", "median", "sum"}
                        else "mean"
                    ),
                )
                n_boot = st.number_input(
                    "Bootstrap draws",
                    min_value=100,
                    max_value=5000,
                    value=int(defaults.get("n_boot", 1000)),
                    step=100,
                )
                ci = st.slider(
                    "Confidence level",
                    min_value=0.5,
                    max_value=0.99,
                    value=float(defaults.get("ci", 0.95)),
                    step=0.01,
                )
                ascending = st.checkbox(
                    "Lower scores are better",
                    value=bool(defaults.get("ascending", True)),
                )
                submitted = st.form_submit_button("Bootstrap rankings")

            if submitted:
                with st.spinner("Bootstrapping ranks..."):
                    try:
                        result = bootstrap_rank_confidence(
                            rank_df,
                            id_col=id_col,
                            score_col=score_col,
                            replicate_col=replicate_col,
                            n_boot=int(n_boot),
                            ci=float(ci),
                            aggfunc=aggfunc,
                            ascending=ascending,
                        )
                    except Exception as exc:
                        st.error(f"Failed to bootstrap rankings: {exc}")
                    else:
                        st.session_state["power_rank_ci_result"] = result
                        st.session_state["power_rank_ci_form"] = {
                            "id_col": id_col,
                            "score_col": score_col,
                            "replicate_col": replicate_col,
                            "aggfunc": aggfunc,
                            "n_boot": int(n_boot),
                            "ci": float(ci),
                            "ascending": ascending,
                        }

        rank_ci_result = st.session_state.get("power_rank_ci_result")
        if isinstance(rank_ci_result, pd.DataFrame) and not rank_ci_result.empty:
            st.dataframe(rank_ci_result, hide_index=True, use_container_width=True)

    with power_tabs[3]:
        st.write("Measure per-signature stability across replicate profiles.")
        upload = st.file_uploader(
            "Replicate-level gene scores (parquet/csv/tsv)",
            type=["csv", "tsv", "txt", "parquet", "pq", "json", "jsonl"],
            key="power_stability_upload",
        )
        stability_df: Optional[pd.DataFrame] = None
        if upload is not None:
            try:
                stability_df = read_uploaded_table(upload)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state["power_stability_table"] = stability_df
        elif "power_stability_table" in st.session_state:
            stability_df = st.session_state.get("power_stability_table")

        if isinstance(stability_df, pd.DataFrame):
            st.dataframe(stability_df.head(20), hide_index=True, use_container_width=True)
            defaults = st.session_state.get("power_stability_form", {})
            with st.form("power_stability_form"):
                signature_col = st.text_input(
                    "Signature column",
                    value=defaults.get("signature_col", "signature_id"),
                )
                replicate_col = st.text_input(
                    "Replicate column",
                    value=defaults.get("replicate_col", "replicate_id"),
                )
                gene_col = st.text_input(
                    "Gene column",
                    value=defaults.get("gene_col", "gene_symbol"),
                )
                score_col = st.text_input(
                    "Score column",
                    value=defaults.get("score_col", "score"),
                )
                method_options = ["spearman", "pearson", "kendall", "cosine"]
                selected_method = defaults.get("method", "spearman")
                if selected_method not in method_options:
                    selected_method = "spearman"
                method = st.selectbox(
                    "Correlation method",
                    method_options,
                    index=method_options.index(selected_method),
                )
                submitted = st.form_submit_button("Compute stability")

            if submitted:
                with st.spinner("Computing stability metrics..."):
                    try:
                        result = compute_signature_stability(
                            stability_df,
                            signature_col=signature_col,
                            replicate_col=replicate_col,
                            gene_col=gene_col,
                            score_col=score_col,
                            method=method,
                        )
                    except Exception as exc:
                        st.error(f"Failed to compute stability: {exc}")
                    else:
                        st.session_state["power_stability_result"] = result
                        st.session_state["power_stability_form"] = {
                            "signature_col": signature_col,
                            "replicate_col": replicate_col,
                            "gene_col": gene_col,
                            "score_col": score_col,
                            "method": method,
                        }

        stability_result = st.session_state.get("power_stability_result")
        if isinstance(stability_result, pd.DataFrame) and not stability_result.empty:
            st.dataframe(stability_result, hide_index=True, use_container_width=True)

    with power_tabs[4]:
        st.write("Estimate false discovery rates by permuting hit labels.")
        source = st.radio(
            "Data source",
            ["Current results", "Upload table"],
            index=0,
            horizontal=True,
            key="power_fdr_source",
        )
        fdr_df: Optional[pd.DataFrame] = None
        if source == "Current results":
            if ranking_df is not None and not ranking_df.empty:
                fdr_df = ranking_df
            else:
                st.info("Run scoring to generate rankings or upload a table.")
        else:
            upload = st.file_uploader(
                "Ranked table (parquet/csv/tsv)",
                type=["csv", "tsv", "txt", "parquet", "pq", "json", "jsonl"],
                key="power_fdr_upload",
            )
            if upload is not None:
                try:
                    fdr_df = read_uploaded_table(upload)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.session_state["power_fdr_table"] = fdr_df
            elif "power_fdr_table" in st.session_state:
                fdr_df = st.session_state.get("power_fdr_table")

        if isinstance(fdr_df, pd.DataFrame):
            defaults = st.session_state.get("power_fdr_form", {})
            label_default = defaults.get("label_col", "is_hit")
            score_default = defaults.get("score_col", "score")
            with st.form("power_fdr_form"):
                score_col = st.text_input("Score column", value=score_default)
                label_col = st.text_input("Hit label column", value=label_default)
                top_k = st.number_input(
                    "Top K",
                    min_value=1,
                    max_value=max(1, int(len(fdr_df))),
                    value=int(defaults.get("top_k", min(50, len(fdr_df)))),
                    step=1,
                )
                n_sim = st.number_input(
                    "Permutations",
                    min_value=100,
                    max_value=10000,
                    value=int(defaults.get("n_sim", 2000)),
                    step=100,
                )
                higher_is_better = st.checkbox(
                    "Higher scores signify stronger hits",
                    value=bool(defaults.get("higher_is_better", False)),
                )
                submitted = st.form_submit_button("Simulate FDR")

            if submitted:
                if label_col not in fdr_df.columns:
                    st.error(f"Column '{label_col}' not found in table.")
                elif score_col not in fdr_df.columns:
                    st.error(f"Column '{score_col}' not found in table.")
                else:
                    with st.spinner("Running permutations..."):
                        try:
                            res = simulate_false_discovery_rate(
                                fdr_df,
                                score_col=score_col,
                                label_col=label_col,
                                top_k=int(top_k),
                                n_sim=int(n_sim),
                                ascending=not higher_is_better,
                            )
                        except Exception as exc:
                            st.error(f"Failed to simulate FDR: {exc}")
                        else:
                            st.session_state["power_fdr_result"] = res
                            st.session_state["power_fdr_form"] = {
                                "score_col": score_col,
                                "label_col": label_col,
                                "top_k": int(top_k),
                                "n_sim": int(n_sim),
                                "higher_is_better": higher_is_better,
                            }

        fdr_result = st.session_state.get("power_fdr_result")
        if isinstance(fdr_result, dict):
            metrics = [
                ("Observed hits", fdr_result.get("observed_hits")),
                ("Expected false", f"{fdr_result.get('expected_false_positives', 0):.2f}"),
                ("Estimated FDR", f"{fdr_result.get('estimated_fdr', 0):.3f}"),
                ("Permutation p-value", f"{fdr_result.get('p_value', 0):.4f}"),
            ]
            cols = st.columns(len(metrics))
            for col, (label, value) in zip(cols, metrics):
                col.metric(label, value)
            with st.expander("Null distribution", expanded=False):
                st.line_chart(fdr_result.get("null_distribution"))

    with power_tabs[5]:
        st.write("Permutation test for comparing two sets of scores.")
        defaults = st.session_state.get("power_permutation_form", {})
        with st.form("power_permutation_form"):
            group_a_text = st.text_area(
                "Group A values",
                value=defaults.get("group_a_text", "0.1, 0.2, 0.18"),
            )
            group_b_text = st.text_area(
                "Group B values",
                value=defaults.get("group_b_text", "0.3, 0.35, 0.4"),
            )
            statistic = st.selectbox(
                "Statistic",
                ["difference_in_means", "difference_in_medians", "cohens_d"],
                index=["difference_in_means", "difference_in_medians", "cohens_d"].index(
                    defaults.get("statistic", "difference_in_means")
                    if defaults.get("statistic", "difference_in_means")
                    in {"difference_in_means", "difference_in_medians", "cohens_d"}
                    else "difference_in_means"
                ),
            )
            n_perm = st.number_input(
                "Permutations",
                min_value=100,
                max_value=10000,
                value=int(defaults.get("n_permutations", 2000)),
                step=100,
            )
            alternative = st.selectbox(
                "Alternative",
                ["two-sided", "greater", "less"],
                index=["two-sided", "greater", "less"].index(
                    defaults.get("alternative", "two-sided")
                    if defaults.get("alternative", "two-sided") in {"two-sided", "greater", "less"}
                    else "two-sided"
                ),
            )
            submitted = st.form_submit_button("Run permutation test")

        if submitted:
            try:
                group_a_vals = parse_float_list(group_a_text)
                group_b_vals = parse_float_list(group_b_text)
            except ValueError as exc:
                st.error(str(exc))
            else:
                if not group_a_vals or not group_b_vals:
                    st.error("Both groups require at least one numeric value.")
                else:
                    with st.spinner("Permuting groups..."):
                        try:
                            res = permutation_significance_test(
                                group_a_vals,
                                group_b_vals,
                                statistic=statistic,
                                n_permutations=int(n_perm),
                                alternative=alternative,
                            )
                        except Exception as exc:
                            st.error(f"Failed to run permutation test: {exc}")
                        else:
                            st.session_state["power_permutation_result"] = res
                            st.session_state["power_permutation_form"] = {
                                "group_a_text": group_a_text,
                                "group_b_text": group_b_text,
                                "statistic": statistic,
                                "n_permutations": int(n_perm),
                                "alternative": alternative,
                            }

        perm_result = st.session_state.get("power_permutation_result")
        if isinstance(perm_result, dict):
            st.metric("Observed statistic", f"{perm_result.get('observed_statistic', 0):.4f}")
            st.metric("Permutation p-value", f"{perm_result.get('p_value', 0):.4f}")
            with st.expander("Null distribution", expanded=False):
                st.line_chart(perm_result.get("null_distribution"))

    bookmark_payload = {
        "version": SESSION_VERSION,
        "library_path": str(selected_library_path),
        "method": method,
        "top_k": int(top_k),
        "blend": (
            float(blend) if (blend is not None and method == "metric") else None
        ),
        "cell_line_filter": cln,
        "target_context": target_context if isinstance(target_context, dict) else {},
        "target_signature": target_sig.model_dump(),
        "model_checkpoint": ({"label": model_label} if model_label else None),
    }
    bookmark_token = encode_state_token(bookmark_payload)
    st.session_state["bookmark_token"] = bookmark_token

    with st.sidebar.expander("Session & sharing", expanded=False):
        if session_bytes:
            st.download_button(
                "Export session JSON",
                data=session_bytes,
                file_name="scperturb_cmap_session.json",
                mime="application/json",
                width="stretch",
            )
        session_import = st.file_uploader(
            "Import session JSON", type="json", key="session_import"
        )
        if session_import is not None:
            try:
                payload = json.loads(session_import.getvalue().decode("utf-8"))
                apply_state_payload(payload, include_results=True)
                st.toast("Session imported", icon="📄")
                st.experimental_rerun()
            except Exception as exc:
                st.error(f"Failed to import session: {exc}")
        st.text_input(
            "Bookmark token",
            value=bookmark_token,
            key="bookmark_token_display",
            help=f"Share or append to the URL as ?{BOOKMARK_PARAM}=<token>",
            disabled=True,
        )
        if st.button("Apply bookmark to URL", key="bookmark_update"):
            st.query_params.update({BOOKMARK_PARAM: bookmark_token})
            st.toast("Bookmark added to browser URL", icon="🔗")
        st.code(f"?{BOOKMARK_PARAM}={bookmark_token}")

    # MOA enrichment visuals will be updated below once results are available
    ranking_ready = ranking_df is not None and not ranking_df.empty
    if ranking_ready:
        e_df = moa_enrichment(ranking_df, top_n=50)
        st.plotly_chart(plot_moa_enrichment_bar(e_df), width="stretch")
        st.plotly_chart(
            plot_moa_enrichment_heatmap(ranking_df, e_df),
            width="stretch",
        )


if __name__ == "__main__":
    main()
