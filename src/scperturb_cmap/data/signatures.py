from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

try:
    import scipy.sparse as sp
    from scipy.stats import rankdata
except Exception:  # pragma: no cover - optional vectorized rank support
    sp = None  # type: ignore
    rankdata = None  # type: ignore

from scperturb_cmap.data.preprocess import standardize_vector
from scperturb_cmap.io.schemas import TargetSignature


def _to_numpy(X) -> np.ndarray:
    if sp is not None and hasattr(sp, "issparse") and sp.issparse(X):
        return X.toarray()
    # anndata may store as numpy.matrix; ensure 2D ndarray
    return np.asarray(X)


def _rank_biserial_weights(group1: np.ndarray, group2: np.ndarray) -> np.ndarray:
    n1, n2 = group1.shape[0], group2.shape[0]
    weights = np.zeros(group1.shape[1], dtype=float)
    if n1 == 0 or n2 == 0:
        return weights
    if rankdata is None:
        # Fallback to Cohen's d-like effect size if SciPy unavailable
        mu1 = group1.mean(axis=0)
        mu2 = group2.mean(axis=0)
        s1 = group1.std(axis=0, ddof=0)
        s2 = group2.std(axis=0, ddof=0)
        pooled = np.sqrt((s1**2 + s2**2) / 2.0)
        eps = max(np.finfo(float).eps, 1e-12)
        pooled = np.where(pooled < eps, eps, pooled)
        return (mu1 - mu2) / pooled

    # Compute per-gene rank-biserial via Mann-Whitney U relationship
    for j in range(group1.shape[1]):
        x = group1[:, j]
        y = group2[:, j]
        z = np.concatenate([x, y], axis=0)
        r = rankdata(z, method="average")
        R1 = r[:n1].sum()
        U1 = R1 - n1 * (n1 + 1) / 2.0
        rb = 2.0 * U1 / (n1 * n2) - 1.0
        weights[j] = rb
    return weights


def _select_cells(adata, barcodes: Sequence[str]) -> np.ndarray:
    obs_index = np.array(adata.obs_names)
    mask = np.isin(obs_index, np.array(list(barcodes)))
    return mask


def target_from_cells(
    adata,
    target_barcodes: List[str],
    ref_barcodes: Optional[List[str]] = None,
    method: str = "rank_biserial",
) -> TargetSignature:
    """Construct a TargetSignature from selected cells.

    - method="rank_biserial": per-gene rank-biserial (fallback to Cohen's d)
    - weights standardized; gene order follows adata.var_names
    - only genes present in adata.var_names are returned
    """
    target_mask = _select_cells(adata, target_barcodes)
    if ref_barcodes is None:
        ref_mask = ~target_mask
    else:
        ref_mask = _select_cells(adata, ref_barcodes)

    if not target_mask.any():
        raise ValueError("No target cells selected")
    if not ref_mask.any():
        raise ValueError("No reference cells selected")

    X = _to_numpy(adata.X)
    X = np.asarray(X, dtype=float)

    g1 = X[target_mask, :]
    g2 = X[ref_mask, :]

    if method == "rank_biserial":
        weights = _rank_biserial_weights(g1, g2)
    elif method.lower() in {"d", "cohen_d", "cohen-d", "effect"}:
        mu1 = g1.mean(axis=0)
        mu2 = g2.mean(axis=0)
        s1 = g1.std(axis=0, ddof=0)
        s2 = g2.std(axis=0, ddof=0)
        pooled = np.sqrt((s1**2 + s2**2) / 2.0)
        eps = max(np.finfo(float).eps, 1e-12)
        pooled = np.where(pooled < eps, eps, pooled)
        weights = (mu1 - mu2) / pooled
    elif method.lower() in {"logfc", "log_fc", "delta"}:
        # Simple difference in means (data expected already on log scale if desired)
        weights = g1.mean(axis=0) - g2.mean(axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    weights = standardize_vector(weights)
    genes = list(map(str, list(adata.var_names)))

    return TargetSignature(genes=genes, weights=weights.tolist(), metadata={"method": method})


def target_from_cluster(
    adata,
    cluster_key: str,
    cluster: str,
    reference: str = "rest",
    method: str = "rank_biserial",
) -> TargetSignature:
    """Construct a TargetSignature comparing a cluster to reference (default: rest)."""
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"cluster_key '{cluster_key}' not in adata.obs")
    labels = adata.obs[cluster_key].astype(str)
    target_barcodes = labels.index[labels == str(cluster)].tolist()

    if reference == "rest":
        ref_barcodes = labels.index[labels != str(cluster)].tolist()
    else:
        # Allow specifying another label as reference
        ref_barcodes = labels.index[labels == str(reference)].tolist()

    return target_from_cells(
        adata,
        target_barcodes=target_barcodes,
        ref_barcodes=ref_barcodes,
        method=method,
    )


def target_from_gene_lists(up_genes: List[str], down_genes: List[str]) -> TargetSignature:
    """Create a TargetSignature from explicit up/down gene lists.

    Weights are +1 for up and -1 for down, then standardized.
    """
    genes: List[str] = []
    weights: List[float] = []
    for g in up_genes:
        genes.append(str(g))
        weights.append(1.0)
    for g in down_genes:
        genes.append(str(g))
        weights.append(-1.0)

    w = standardize_vector(np.asarray(weights, dtype=float))
    return TargetSignature(genes=genes, weights=w.tolist(), metadata={"method": "gene_lists"})
