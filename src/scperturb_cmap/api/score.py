from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from scperturb_cmap.data.lincs_loader import pivot_signatures
from scperturb_cmap.data.preprocess import harmonize_symbols, standardize_vector
from scperturb_cmap.io.schemas import ScoreResult, TargetSignature
from scperturb_cmap.models.baseline import (
    cosine_connectivity,
    ensemble_connectivity,
    gsea_connectivity,
)
from scperturb_cmap.models.dual_encoder import DualEncoder

logger = logging.getLogger(__name__)
LibraryType = Union[
    pd.DataFrame,  # long form with gene_symbol/score
    Tuple[np.ndarray, List[str], pd.DataFrame],  # (matrix, genes, meta)
    Dict[str, Any],  # {"matrix": ndarray, "genes": list[str], "meta": DataFrame}
]


def _as_pivot(library: LibraryType) -> Tuple[np.ndarray, List[str], pd.DataFrame, pd.DataFrame]:
    """Return (M, genes, meta, long_df) from various library inputs.

    long_df is returned if available for GSEA; otherwise reconstructed from pivot.
    """
    if isinstance(library, pd.DataFrame):
        long_df = library
        M, genes, meta = pivot_signatures(long_df)
        return M, genes, meta, long_df

    if isinstance(library, tuple) and len(library) == 3:
        M, genes, meta = library  # type: ignore[misc]
        if not isinstance(meta, pd.DataFrame):
            raise TypeError("meta must be a pandas.DataFrame")
        # Build a minimal long df from pivot for GSEA
        long_df = pd.DataFrame(
            {
                "signature_id": np.repeat(meta["signature_id"].to_numpy(), len(genes)),
                "compound": np.repeat(meta["compound"].to_numpy(), len(genes)),
                "cell_line": np.repeat(meta["cell_line"].to_numpy(), len(genes)),
                "gene_symbol": np.tile(genes, M.shape[0]),
                "score": M.ravel(),
            }
        )
        return M, genes, meta, long_df

    if isinstance(library, dict):
        M = np.asarray(library["matrix"], dtype=float)
        genes = list(map(str, library["genes"]))
        if isinstance(library["meta"], pd.DataFrame):
            meta = library["meta"]
        else:
            meta = pd.DataFrame(library["meta"])  # type: ignore[arg-type]
        long_df = pd.DataFrame(
            {
                "signature_id": np.repeat(meta["signature_id"].to_numpy(), len(genes)),
                "compound": np.repeat(meta["compound"].to_numpy(), len(genes)),
                "cell_line": np.repeat(meta["cell_line"].to_numpy(), len(genes)),
                "gene_symbol": np.tile(genes, M.shape[0]),
                "score": M.ravel(),
            }
        )
        return M, genes, meta, long_df

    raise TypeError("Unsupported library type; expected DataFrame, (matrix, genes, meta), or dict")


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    mu = x.mean()
    sd = x.std(ddof=0)
    eps = max(np.finfo(float).eps, 1e-12)
    if sd < eps:
        return np.zeros_like(x)
    return (x - mu) / sd


def _metric_scores(
    target: TargetSignature,
    M: np.ndarray,
    genes: List[str],
    model_path: str,
) -> np.ndarray:
    # Align target to library genes
    t_genes = harmonize_symbols(target.genes)
    g_ref = harmonize_symbols(genes)
    t_vals = np.asarray(target.weights, dtype=float)
    # Create aligned target vector
    ref_vec = np.asarray(t_vals)
    # Build mapping for first occurrences
    seen: set[str] = set()
    idx_map: dict[str, int] = {}
    for i, g in enumerate(t_genes):
        if g in seen:
            continue
        seen.add(g)
        idx_map[g] = i
    # Aligned order same as library genes
    aligned = np.zeros(len(g_ref), dtype=float)
    for j, g in enumerate(g_ref):
        if g in idx_map:
            aligned[j] = ref_vec[idx_map[g]]
    aligned = standardize_vector(aligned)

    # Load model
    ckpt = torch.load(model_path, map_location="cpu")
    input_dim = int(ckpt.get("config", {}).get("input_dim", len(genes)))
    model = DualEncoder(input_dim=input_dim, embed_dim=64)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        # Adapt feature dimension if model input dimension differs from library genes
        if aligned.shape[0] < input_dim:
            pad = np.zeros(input_dim - aligned.shape[0], dtype=aligned.dtype)
            left_vec = np.concatenate([aligned, pad], axis=0)
        else:
            left_vec = aligned[:input_dim]

        left = torch.tensor(left_vec, dtype=torch.float32).unsqueeze(0)
        zL, _, _ = model(left, left)
        zL = zL / (zL.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        # Right embeddings for all rows
        if M.shape[1] < input_dim:
            padR = np.zeros((M.shape[0], input_dim - M.shape[1]), dtype=M.dtype)
            R_np = np.concatenate([M, padR], axis=1)
        else:
            R_np = M[:, :input_dim]
        R = torch.tensor(R_np, dtype=torch.float32)
        _, zR, _ = model(R, R)
        zR = zR / (zR.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        sim = (zR @ zL.squeeze(0))  # shape [num_signatures]
    # Lower is better (negative implies inversion)
    return sim.numpy()


def rank_drugs(
    target_signature: TargetSignature,
    library: LibraryType,
    method: str = "baseline",
    model_path: Optional[str] = None,
    top_k: int = 50,
    blend: float = 0.5,
    auto_blend: bool = False,
) -> ScoreResult:
    M, genes, meta, long_df = _as_pivot(library)

    # Guardrail: minimum overlap if target is large
    t_genes = harmonize_symbols(target_signature.genes)
    lib_genes = harmonize_symbols(genes)
    overlap = len(set(t_genes) & set(lib_genes))
    if len(t_genes) >= 300 and overlap < 150:
        missing = [g for g in t_genes if g not in set(lib_genes)][:10]
        hint = (
            "Insufficient gene overlap. Consider mapping symbols (validate-h5ad), "
            "supplying --library-genes to make-target for QC, or restricting to "
            "L1000 landmarks."
        )
        msg = (
            f"Insufficient gene overlap: {overlap} < 150. "
            f"Example missing target genes: {missing}. {hint}"
        )
        raise ValueError(msg)

    # Baseline ensemble (lower is better)
    cos_df = cosine_connectivity(target_signature, M, genes, meta)
    try:
        gsea_df = gsea_connectivity(target_signature, long_df)
        base_df = ensemble_connectivity(cos_df, gsea_df)
    except Exception as e:
        logger.warning("GSEA connectivity failed (%s); falling back to cosine only.", e)
        base_df = cos_df.copy()

    def _attach_moa_target(df: pd.DataFrame) -> pd.DataFrame:
        cols = ["signature_id"]
        if "moa" in long_df.columns:
            cols.append("moa")
        if "target" in long_df.columns:
            cols.append("target")
        if len(cols) > 1:
            extra = long_df[cols].drop_duplicates(subset=["signature_id"], keep="first")
            out = df.merge(extra, on="signature_id", how="left")
        else:
            out = df.copy()
            out["moa"] = pd.NA
            out["target"] = pd.NA
        return out

    if method == "baseline":
        ranking = base_df.sort_values("score", ascending=True).head(top_k)
        ranking = _attach_moa_target(ranking)
        if not ranking.empty:
            ranking = ranking.copy()
            z_scores = _zscore(ranking["score"].to_numpy())
            ranking["z_score"] = z_scores
            ranking["p_value"] = 2 * norm.sf(np.abs(z_scores))
        # Append FDR q-values (BH) for convenience
        if not ranking.empty and "p_value" in ranking.columns:
            p = ranking["p_value"].astype(float).to_numpy()
            m = max(1, len(p))
            order = pd.Series(p).sort_values().index.to_numpy()
            ranks = pd.Series(range(1, m + 1), index=order).sort_index().to_numpy()
            q = pd.Series(p * m / ranks).clip(upper=1.0)
            q_sorted = q.to_numpy()[order]
            for i in range(m - 2, -1, -1):
                q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
            q_final = pd.Series(index=order, data=q_sorted).sort_index().to_numpy()
            ranking["q_value"] = q_final
        return ScoreResult(method="baseline", ranking=ranking, metadata={"top_k": top_k})

    if method == "metric":
        if model_path is None:
            raise ValueError("model_path is required for method='metric'")
        metric = _metric_scores(target_signature, M, genes, model_path)
        # Blend z-scored baseline and metric
        z_base = _zscore(base_df["score"].to_numpy())
        z_metric = _zscore(metric)
        if auto_blend:
            xb = z_base
            xm = z_metric
            num = -np.dot(xm - xb, xb)
            den = np.dot(xm - xb, xm - xb) + 1e-12
            alpha = float(np.clip(num / den, 0.0, 1.0))
        else:
            alpha = float(blend)
        score = (1.0 - alpha) * z_base + alpha * z_metric
        df = base_df.copy()
        df["score"] = score
        ranking = df.sort_values("score", ascending=True).head(top_k)
        ranking = _attach_moa_target(ranking)
        if not ranking.empty:
            ranking = ranking.copy()
            z_scores = _zscore(ranking["score"].to_numpy())
            ranking["z_score"] = z_scores
            ranking["p_value"] = 2 * norm.sf(np.abs(z_scores))
        # Append FDR q-values (BH)
        if not ranking.empty and "p_value" in ranking.columns:
            p = ranking["p_value"].astype(float).to_numpy()
            m = max(1, len(p))
            order = pd.Series(p).sort_values().index.to_numpy()
            ranks = pd.Series(range(1, m + 1), index=order).sort_index().to_numpy()
            q = pd.Series(p * m / ranks).clip(upper=1.0)
            q_sorted = q.to_numpy()[order]
            for i in range(m - 2, -1, -1):
                q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
            q_final = pd.Series(index=order, data=q_sorted).sort_index().to_numpy()
            ranking["q_value"] = q_final
        return ScoreResult(
            method="metric",
            ranking=ranking,
            metadata={
                "top_k": top_k,
                "blend": (alpha if auto_blend else blend),
                "auto_blend": bool(auto_blend),
                "model_path": model_path,
            },
        )

    raise ValueError("Unknown method; expected 'baseline' or 'metric'")
