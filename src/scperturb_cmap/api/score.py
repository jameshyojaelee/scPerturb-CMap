from __future__ import annotations

import logging
from pathlib import Path
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
    str,
    Path,
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


def _normalize_filter_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if str(v)]


def _filter_dataframe(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    out = df
    for key, col in [("cell_line", "cell_line"), ("moa", "moa"), ("signature_id", "signature_id")]:
        values = _normalize_filter_values(filters.get(key))
        if values and col in out.columns:
            out = out[out[col].astype(str).isin(set(values))]
    return out


def _load_parquet_filtered(path: Path, filters: Dict[str, Any]) -> pd.DataFrame:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError("pyarrow is required to read Parquet datasets") from exc

    dataset = ds.dataset(str(path), format="parquet")
    exprs = []
    names = set(dataset.schema.names)
    for key, col in [("cell_line", "cell_line"), ("moa", "moa"), ("signature_id", "signature_id")]:
        values = _normalize_filter_values(filters.get(key))
        if values and col in names:
            exprs.append(ds.field(col).isin(sorted(set(values))))
    filt = None
    if exprs:
        filt = exprs[0]
        for e in exprs[1:]:
            filt = filt & e
    scanner = dataset.scanner(filter=filt) if filt is not None else dataset.scanner()
    table = scanner.to_table()
    df = table.to_pandas()
    # If partition columns are pruned by the scanner, reattach filtered values so downstream
    # pivots retain cell line metadata.
    for key, col in [("cell_line", "cell_line"), ("moa", "moa")]:
        values = _normalize_filter_values(filters.get(key))
        if values and col not in df.columns:
            df[col] = values[0]
    logger.info(
        "Loaded %s rows from %s (filters=%s)",
        f"{len(df):,}",
        path,
        {k: v for k, v in filters.items() if v},
    )
    return df


def _load_library_with_filters(library: LibraryType | str | Path, filters: Optional[Dict[str, Any]]) -> LibraryType:
    if isinstance(library, (str, Path)):
        return _load_parquet_filtered(Path(library), filters or {})
    if isinstance(library, pd.DataFrame):
        return _filter_dataframe(library.copy(), filters or {})
    return library


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
        # Training uses negated LINCS signatures; mirror that at inference time.
        R_np = -R_np
        R = torch.tensor(R_np, dtype=torch.float32)
        _, zR, _ = model(R, R)
        zR = zR / (zR.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        sim = (zR @ zL.squeeze(0))  # shape [num_signatures]
    # Lower is better (more negative implies stronger inversion)
    return (-sim.numpy())


def rank_drugs(
    target_signature: TargetSignature,
    library: LibraryType,
    method: str = "baseline",
    model_path: Optional[str] = None,
    top_k: int = 50,
    blend: float = 0.5,
    auto_blend: bool = False,
    filters: Optional[Dict[str, Any]] = None,
) -> ScoreResult:
    library = _load_library_with_filters(library, filters)
    if isinstance(library, pd.DataFrame) and len(library) > 1_000_000:
        logger.warning(
            "Scoring with a large library (%s rows); consider applying filters or partitioning.",
            f"{len(library):,}",
        )
    M, genes, meta, long_df = _as_pivot(library)

    t_genes = harmonize_symbols(target_signature.genes)
    lib_genes = harmonize_symbols(genes)
    overlap = len(set(t_genes) & set(lib_genes))
    overlap_fraction = overlap / max(1, len(t_genes))
    warn_cutoff = min(len(t_genes), max(5, int(0.2 * len(t_genes))))
    overlap_warning = overlap < warn_cutoff
    if overlap == 0:
        examples = ", ".join(sorted(set(t_genes))[:3])
        hint = (
            f"Insufficient gene overlap: 0 of {len(t_genes)} target genes matched the library. "
            f"Examples missing: {examples}. "
            "Map gene symbols/aliases or restrict to L1000 landmarks."
        )
        raise ValueError(hint)
    overlap_meta = {
        "overlap_genes": int(overlap),
        "target_genes": int(len(t_genes)),
        "overlap_fraction": float(overlap_fraction),
        "overlap_warning": bool(overlap_warning),
    }

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
        meta_out = {"top_k": top_k, **overlap_meta}
        if filters:
            meta_out["filters"] = filters
        return ScoreResult(method="baseline", ranking=ranking, metadata=meta_out)

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
        meta_out = {
            "method": "metric",
            "ranking": ranking,
            "metadata": {
                "top_k": top_k,
                "blend": (alpha if auto_blend else blend),
                "auto_blend": bool(auto_blend),
                "model_path": model_path,
                **overlap_meta,
            },
        }
        if filters:
            meta_out["metadata"]["filters"] = filters
        return ScoreResult(**meta_out)

    raise ValueError("Unknown method; expected 'baseline' or 'metric'")
