from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from scperturb_cmap.data.preprocess import harmonize_symbols
from scperturb_cmap.io.schemas import TargetSignature


def _row_standardize(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    if M.ndim == 1:
        M = M.reshape(1, -1)
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    eps = max(np.finfo(float).eps, 1e-12)
    out = (M - mu) / np.where(sd < eps, 1.0, sd)
    out[sd[:, 0] < eps] = 0.0
    return out


def cosine_connectivity(
    target: TargetSignature,
    lincs_matrix: np.ndarray,
    lincs_genes: List[str],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """Cosine-based connectivity where lower scores indicate stronger concordance.

    - Aligns genes between target and LINCS genes using harmonized symbols
    - Standardizes vectors (row-wise for LINCS, 1D for target)
    - Computes cosine similarity and returns score = -cosine
    """
    M = np.asarray(lincs_matrix, dtype=float)
    if M.shape[0] != len(meta):
        raise ValueError("meta rows must match lincs_matrix rows")
    genes_lincs = harmonize_symbols(lincs_genes)

    t_genes = harmonize_symbols(target.genes)
    t_weights = np.asarray(target.weights, dtype=float)

    # Build first-occurrence index maps
    lmap = {}
    for j, g in enumerate(genes_lincs):
        if g not in lmap:
            lmap[g] = j
    seen = set()
    t_pairs: List[Tuple[str, float]] = []
    for g, w in zip(t_genes, t_weights):
        if g in seen:
            continue
        seen.add(g)
        if g in lmap:
            t_pairs.append((g, w))
    if not t_pairs:
        # No overlap; assign zeros
        scores = np.zeros(M.shape[0], dtype=float)
    else:
        common_genes = [g for g, _ in t_pairs]
        t_vals = np.array([w for _, w in t_pairs], dtype=float)
        cols = [lmap[g] for g in common_genes]
        M_sub = M[:, cols]

        # Standardize
        Mz = _row_standardize(M_sub)
        tz = _row_standardize(t_vals).ravel()

        # Cosine similarity
        eps = max(np.finfo(float).eps, 1e-12)
        t_norm = np.linalg.norm(tz) + eps
        r_norms = np.linalg.norm(Mz, axis=1) + eps
        cos = (Mz @ tz) / (r_norms * t_norm)
        scores = -cos

    out = meta.copy().reset_index(drop=True)
    out["score"] = scores
    return out


def _gsea_es(ranked_genes: List[str], gene_set: set[str]) -> float:
    N = len(ranked_genes)
    if N == 0:
        return 0.0
    hits = [g in gene_set for g in ranked_genes]
    Nh = sum(hits)
    if Nh == 0:
        return 0.0
    Nm = N - Nh
    phit = 1.0 / Nh
    pmiss = 1.0 / Nm if Nm > 0 else 0.0
    running = 0.0
    best = 0.0
    worst = 0.0
    for h in hits:
        if h:
            running += phit
        else:
            running -= pmiss
        if running > best:
            best = running
        if running < worst:
            worst = running
    return best if abs(best) >= abs(worst) else worst


def gsea_connectivity(
    target: TargetSignature,
    lincs_long: pd.DataFrame,
    top_k: int = 150,
) -> pd.DataFrame:
    """GSEA-style connectivity: positive when up genes enriched at top and down at bottom.

    Returns a score in [-1, 1] per signature.
    """
    required = {"signature_id", "compound", "cell_line", "gene_symbol", "score"}
    missing = sorted(required - set(lincs_long.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    t_genes = harmonize_symbols(target.genes)
    t_weights = np.asarray(target.weights, dtype=float)
    up = {g for g, w in zip(t_genes, t_weights) if w > 0}
    down = {g for g, w in zip(t_genes, t_weights) if w < 0}

    df = lincs_long.copy()
    df["gsym"] = df["gene_symbol"].astype(str).str.strip().str.upper()

    # Optionally limit to top_k highest |score| genes per signature to speed up
    if top_k is not None and top_k > 0:
        df["abs_score"] = df["score"].abs()
        df = (
            df.sort_values(["signature_id", "abs_score"], ascending=[True, False])
            .groupby("signature_id", sort=False)
            .head(top_k)
            .drop(columns=["abs_score"])
        )

    records = []
    group_cols = ["signature_id", "compound", "cell_line"]
    for key, grp in df.groupby(group_cols, sort=False):
        # Rank by score descending
        grp_sorted = grp.sort_values("score", ascending=False)
        ranked = grp_sorted["gsym"].tolist()
        es_up = _gsea_es(ranked, up) if up else 0.0
        es_down = _gsea_es(ranked, down) if down else 0.0
        score = 0.5 * (es_up - es_down)
        rec = dict(zip(group_cols, key))
        rec["score"] = score
        records.append(rec)
    return pd.DataFrame.from_records(records, columns=group_cols + ["score"]) if records else (
        pd.DataFrame(columns=group_cols + ["score"])
    )


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = x.mean() if x.size else 0.0
    sd = x.std(ddof=0) if x.size else 1.0
    eps = max(np.finfo(float).eps, 1e-12)
    if sd < eps:
        return np.zeros_like(x)
    return (x - mu) / sd


def ensemble_connectivity(cos_df: pd.DataFrame, gsea_df: pd.DataFrame) -> pd.DataFrame:
    """Average of z-scored methods (lower is better).

    Cosine scores are already lower-is-better. GSEA scores are flipped
    to match this convention before z-scoring.
    """
    keys = ["signature_id", "compound", "cell_line"]
    left = cos_df[keys + ["score"]].rename(columns={"score": "cos_score"})
    right = gsea_df[keys + ["score"]].rename(columns={"score": "gsea_score"})
    df = pd.merge(left, right, on=keys, how="inner")
    if df.empty:
        return pd.DataFrame(columns=keys + ["score"])
    z_cos = _zscore(df["cos_score"].to_numpy())
    z_gsea = _zscore((-df["gsea_score"]).to_numpy())  # flip to lower-is-better
    df["score"] = 0.5 * (z_cos + z_gsea)
    return df[keys + ["score"]]

