from __future__ import annotations

import pandas as pd
from scipy.stats import fisher_exact


def moa_enrichment(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Compute MOA enrichment among top_n compounds vs the rest using Fisher tests.

    Assumes 'moa' and 'score' columns exist. Lower score means stronger inversion.
    Returns DataFrame with columns: ['moa','top','rest','odds_ratio','p_value'].
    """
    if df is None or df.empty or "moa" not in df.columns or "score" not in df.columns:
        return pd.DataFrame(columns=["moa", "top", "rest", "odds_ratio", "p_value"])

    ranked = df.sort_values("score", ascending=True).copy()
    top = ranked.head(int(top_n))
    rest = ranked.iloc[int(top_n) :]

    moa_counts_top = top["moa"].fillna("NA").value_counts()
    moa_counts_rest = rest["moa"].fillna("NA").value_counts()

    rows = []
    all_moas = sorted(set(moa_counts_top.index) | set(moa_counts_rest.index))
    total_top = len(top)
    total_rest = len(rest)
    for m in all_moas:
        a = int(moa_counts_top.get(m, 0))
        b = total_top - a
        c = int(moa_counts_rest.get(m, 0))
        d = total_rest - c
        if (a + b == 0) or (c + d == 0):
            continue
        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append({"moa": m, "top": a, "rest": c, "odds_ratio": odds, "p_value": p})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("p_value", ascending=True).reset_index(drop=True)
    return out

