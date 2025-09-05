from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def harmonize_symbols(genes: Sequence[str]) -> List[str]:
    """Normalize gene symbols.

    - Upper-case
    - Strip surrounding whitespace and remove internal spaces
    - Preserve existing punctuation (e.g., keep "MT-" prefix intact)
    Order and length are preserved; duplicates are not removed.
    """
    out: List[str] = []
    for g in genes:
        if g is None:
            s = ""
        else:
            s = str(g)
        s = s.strip().upper().replace(" ", "")
        out.append(s)
    return out


def _unique_first_with_index(seq: Sequence[str]) -> List[Tuple[str, int]]:
    seen = set()
    pairs: List[Tuple[str, int]] = []
    for i, s in enumerate(seq):
        if s not in seen:
            seen.add(s)
            pairs.append((s, i))
    return pairs


def align_vectors(
    genes_ref: List[str],
    values_ref: np.ndarray,
    genes_query: List[str],
    values_query: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Intersect and align two gene/value vectors.

    - Symbols are harmonized before intersection.
    - For duplicates, the first occurrence in each list is used.
    - The intersected gene list follows the reference gene order.
    Returns (values_ref_aligned, values_query_aligned, common_genes).
    """
    if len(genes_ref) != np.asarray(values_ref).size:
        raise ValueError("genes_ref and values_ref must have the same length")
    if len(genes_query) != np.asarray(values_query).size:
        raise ValueError("genes_query and values_query must have the same length")

    gr = harmonize_symbols(genes_ref)
    gq = harmonize_symbols(genes_query)

    # Indices of first occurrence per gene
    ref_first = dict(_unique_first_with_index(gr))
    qry_first = dict(_unique_first_with_index(gq))

    # Intersection in the order of reference unique genes
    ref_unique_ordered = [g for g, _ in _unique_first_with_index(gr)]
    common_genes = [g for g in ref_unique_ordered if g in qry_first]

    ref_idx = [ref_first[g] for g in common_genes]
    qry_idx = [qry_first[g] for g in common_genes]

    vref = np.asarray(values_ref, dtype=float).ravel()[ref_idx]
    vqry = np.asarray(values_query, dtype=float).ravel()[qry_idx]

    return vref, vqry, common_genes


def standardize_vector(x: np.ndarray) -> np.ndarray:
    """Return a standardized copy of ``x`` (zero mean, unit variance).

    If variance is effectively zero, returns a zero vector to avoid division
    by very small numbers.
    """
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        return arr.copy()
    mu = arr.mean()
    sigma = arr.std(ddof=0)
    eps = max(np.finfo(float).eps, 1e-12)
    if sigma < eps:
        return np.zeros_like(arr)
    return (arr - mu) / sigma

