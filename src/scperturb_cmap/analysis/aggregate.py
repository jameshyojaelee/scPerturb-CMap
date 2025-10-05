from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _modz_weights(mat: pd.DataFrame) -> np.ndarray:
    """Compute MODZ weights for replicate x gene matrix."""
    if mat.shape[0] <= 1:
        return np.ones(mat.shape[0], dtype=float)

    corr = mat.T.corr(method="spearman").fillna(0.0)
    np.fill_diagonal(corr.values, 0.0)
    weights = corr.sum(axis=1).to_numpy(dtype=float)
    weights = np.clip(weights, a_min=0.0, a_max=None)
    if np.allclose(weights.sum(), 0.0):
        return np.ones(mat.shape[0], dtype=float)
    return weights


def collapse_replicates_modz(
    df: pd.DataFrame,
    *,
    signature_col: str = "signature_id",
    replicate_col: str = "replicate_id",
    gene_col: str = "gene_symbol",
    score_col: str = "score",
    metadata_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Collapse replicate measurements using MODZ weighting.

    Returns a new long-form DataFrame with one row per signature/gene.
    If ``replicate_col`` is absent or all values are null, the input is returned unchanged.
    """

    if replicate_col not in df.columns:
        return df.copy()

    if metadata_cols is None:
        metadata_cols = [
            c
            for c in df.columns
            if c not in {replicate_col, gene_col, score_col}
            and c != signature_col
        ]

    out_frames: list[pd.DataFrame] = []
    group_cols = [signature_col]
    for sig, grp in df.groupby(group_cols, sort=False):
        if grp[replicate_col].isna().all():
            out_frames.append(grp.drop(columns=[replicate_col]))
            continue

        pivot = (
            grp.pivot_table(
                index=replicate_col,
                columns=gene_col,
                values=score_col,
                aggfunc="mean",
            )
            .sort_index()
            .fillna(0.0)
        )

        weights = _modz_weights(pivot)
        weights = weights / weights.sum()
        collapsed = np.average(pivot.to_numpy(dtype=float), axis=0, weights=weights)

        collapsed_df = pd.DataFrame(
            {
                signature_col: [sig] * len(pivot.columns),
                gene_col: pivot.columns.to_list(),
                score_col: collapsed,
            }
        )

        if metadata_cols:
            meta = (
                grp[[signature_col, *metadata_cols]]
                .drop_duplicates(subset=[signature_col], keep="first")
            )
            collapsed_df = collapsed_df.merge(meta, on=signature_col, how="left")

        out_frames.append(collapsed_df)

    return pd.concat(out_frames, ignore_index=True)


__all__ = ["collapse_replicates_modz"]
