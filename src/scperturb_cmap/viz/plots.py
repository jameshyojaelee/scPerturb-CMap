from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

EXTERNAL_ID_LINKS = {
    "chembl_id": {
        "label": "ChEMBL",
        "url_template": "https://www.ebi.ac.uk/chembl/compound_report_card/{id}/",
    },
    "drugbank_id": {
        "label": "DrugBank",
        "url_template": "https://go.drugbank.com/drugs/{id}",
    },
    "pubchem_cid": {
        "label": "PubChem",
        "url_template": "https://pubchem.ncbi.nlm.nih.gov/compound/{id}",
    },
    "chebi_id": {
        "label": "ChEBI",
        "url_template": "https://www.ebi.ac.uk/chebi/searchId.do?chebiId={id}",
    },
}


def plot_moa_enrichment_bar(enrich_df: pd.DataFrame, k: int = 12) -> go.Figure:
    """Horizontal bar plot of -log10(p) for top-k enriched MOAs."""
    if enrich_df is None or enrich_df.empty:
        return go.Figure()
    df = enrich_df.copy()
    df["neglog10p"] = -np.log10(df["p_value"].clip(lower=1e-300))
    df = df.head(int(k))
    fig = px.bar(
        df,
        x="neglog10p",
        y="moa",
        orientation="h",
        labels={"neglog10p": "-log10(p)", "moa": "MOA"},
        title="MOA enrichment among top hits",
    )
    fig.update_traces(marker_color="#38bdf8", marker_line_color="#0f172a", marker_line_width=1.2)
    if all(c in df.columns for c in ["top", "rest", "odds_ratio", "p_value"]):
        custom_columns = [
            df["top"].to_numpy(),
            df["rest"].to_numpy(),
            df["odds_ratio"].to_numpy(),
            df["p_value"].to_numpy(),
        ]
        hover_lines = [
            "MOA: %{y}",
            "-log10(p): %{x:.2f}",
            "Top hits: %{customdata[0]}",
            "Background hits: %{customdata[1]}",
            "Odds ratio: %{customdata[2]:.2f}",
            "p-value: %{customdata[3]:.2e}",
        ]
        custom_index = 4
        for col in df.columns:
            key = col.lower()
            if key not in EXTERNAL_ID_LINKS:
                continue
            info = EXTERNAL_ID_LINKS[key]
            ids = df[col].fillna("").astype(str).to_numpy()
            urls = np.array(
                [info["url_template"].format(id=v) if v else "" for v in ids]
            )
            custom_columns.append(ids)
            hover_lines.append(f"{info['label']} ID: %{{customdata[{custom_index}]}}")
            custom_index += 1
            custom_columns.append(urls)
            hover_lines.append(f"{info['label']} URL: %{{customdata[{custom_index}]}}")
            custom_index += 1
        custom_data = np.column_stack(custom_columns)
        hover_template = "<br>".join(hover_lines) + "<extra></extra>"
        fig.update_traces(hovertemplate=hover_template, customdata=custom_data)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.7)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
    )
    return fig


def plot_target_signature_bars(
    genes: Sequence[str], weights: Sequence[float], top_n: int = 20
) -> go.Figure:
    """Bar plot showing top positive and negative genes in the target signature."""
    if genes is None or len(genes) == 0:
        return go.Figure()
    s = pd.Series(weights, index=list(genes), dtype=float).dropna()
    s = s[~s.index.duplicated(keep="first")]
    pos = s.sort_values(ascending=False).head(int(top_n))
    neg = s.sort_values(ascending=True).head(int(top_n))
    df = pd.concat([neg, pos]).reset_index()
    df.columns = ["gene", "weight"]
    df["direction"] = np.where(df["weight"] >= 0, "Up", "Down")
    # order bars by absolute weight
    df = df.iloc[np.argsort(np.abs(df["weight"]).values)]
    fig = px.bar(
        df,
        x="weight",
        y="gene",
        color="direction",
        orientation="h",
        title="Target signature: top positive and negative genes",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=560)
    return fig


def plot_moa_enrichment_heatmap(
    df: pd.DataFrame, enrich_df: pd.DataFrame | None = None
) -> go.Figure:
    """Plot MOA enrichment scores as a heatmap grouped by cell line with rich hover."""
    if df is None or df.empty:
        return go.Figure()
    required = {"cell_line", "moa", "score"}
    if not required.issubset(df.columns):
        return go.Figure()

    pivot = (
        df.pivot_table(index="moa", columns="cell_line", values="score", aggfunc="mean")
        .fillna(0.0)
        .sort_index()
    )
    count_pivot = (
        df.groupby(["moa", "cell_line"], dropna=False).size().unstack(fill_value=0)
        if {"moa", "cell_line"}.issubset(df.columns)
        else pd.DataFrame(index=pivot.index, columns=pivot.columns, data=0)
    )
    count_pivot = count_pivot.reindex(index=pivot.index, columns=pivot.columns, fill_value=0)

    odds_map: dict[str, float] = {}
    pval_map: dict[str, float] = {}
    link_keys: list[str] = []
    link_info: dict[str, dict[str, Any]] = {}
    if enrich_df is not None and not enrich_df.empty:
        for _, row in enrich_df.iterrows():
            moa = row.get("moa")
            if not isinstance(moa, str):
                continue
            if "odds_ratio" in enrich_df.columns:
                odds_map[moa] = float(row.get("odds_ratio", np.nan))
            if "p_value" in enrich_df.columns:
                pval_map[moa] = float(row.get("p_value", np.nan))
        for col in enrich_df.columns:
            key = col.lower()
            if key not in EXTERNAL_ID_LINKS:
                continue
            info = EXTERNAL_ID_LINKS[key]
            link_keys.append(col)
            values = (
                enrich_df[["moa", col]]
                .dropna(subset=["moa"])
                .astype({"moa": str, col: str})
            )
            link_info[col] = {
                "label": info["label"],
                "template": info["url_template"],
                "mapping": dict(values.itertuples(index=False)),
            }

    rows, cols = pivot.shape
    custom_depth = 3 + 2 * len(link_keys)
    custom_data = np.empty((rows, cols, max(custom_depth, 1)), dtype=object)
    custom_data[:] = ""

    odds_strings = [
        "" if not np.isfinite(odds_map.get(moa, np.nan)) else f"{odds_map[moa]:.2f}"
        for moa in pivot.index
    ]
    pval_strings = [
        "" if not np.isfinite(pval_map.get(moa, np.nan)) else f"{pval_map[moa]:.2e}"
        for moa in pivot.index
    ]

    for i, moa in enumerate(pivot.index):
        for j, cell in enumerate(pivot.columns):
            count_val = int(count_pivot.iloc[i, j])
            custom_data[i, j, 0] = count_val
            custom_data[i, j, 1] = odds_strings[i]
            custom_data[i, j, 2] = pval_strings[i]
            offset = 3
            for col in link_keys:
                label_value = link_info[col]["mapping"].get(moa, "")
                url_value = (
                    link_info[col]["template"].format(id=label_value)
                    if label_value
                    else ""
                )
                custom_data[i, j, offset] = label_value
                custom_data[i, j, offset + 1] = url_value
                offset += 2

    hover_lines = [
        "MOA: %{y}",
        "Cell line: %{x}",
        "Mean score: %{z:.3f}",
        "Top hits: %{customdata[0]}",
        "Odds ratio: %{customdata[1]}",
        "p-value: %{customdata[2]}",
    ]
    base_offset = 3
    for idx, col in enumerate(link_keys):
        label = link_info[col]["label"]
        hover_lines.append(f"{label} ID: %{{customdata[{base_offset + idx * 2}]}}")
        hover_lines.append(
            f"{label} URL: %{{customdata[{base_offset + idx * 2 + 1}]}}"
        )
    hover_template = "<br>".join(hover_lines) + "<extra></extra>"

    heatmap = go.Heatmap(
        z=pivot.to_numpy(),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.0, "#ef4444"],
            [0.5, "#1f2937"],
            [1.0, "#22d3ee"],
        ],
        colorbar_title="Mean score",
        customdata=custom_data,
        hovertemplate=hover_template,
    )
    fig = go.Figure(data=heatmap)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.7)",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
