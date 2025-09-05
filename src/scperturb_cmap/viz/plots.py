from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420)
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

