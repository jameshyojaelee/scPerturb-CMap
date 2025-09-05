from __future__ import annotations

import numpy as np

from scperturb_cmap.data.preprocess import (
    align_vectors,
    harmonize_symbols,
    standardize_vector,
)


def test_harmonize_symbols_order_and_format():
    genes = ["  gapdh ", "mt-CO1", "B Raf", "TP53"]
    out = harmonize_symbols(genes)
    assert out == ["GAPDH", "MT-CO1", "BRAF", "TP53"]


def test_align_vectors_with_duplicates_and_order_invariance():
    # Reference has duplicates for A
    genes_ref = ["A", "B", "A", "C"]
    values_ref = np.array([1.0, 2.0, 3.0, 4.0])

    # Query has duplicates for C; different order, different casing
    genes_query = ["c", "a", "C", "b"]
    values_query = np.array([40.0, 10.0, 30.0, 20.0])

    vref, vqry, common = align_vectors(genes_ref, values_ref, genes_query, values_query)
    # Common genes follow reference unique order: A, B, C
    assert common == ["A", "B", "C"]
    # First occurrences used for duplicates
    assert np.allclose(vref, np.array([1.0, 2.0, 4.0]))
    assert np.allclose(vqry, np.array([10.0, 20.0, 40.0]))

    # Reordering query should not change aligned outputs
    genes_query2 = ["b", "c", "a", "c"]
    values_query2 = np.array([20.0, 40.0, 10.0, 30.0])
    vref2, vqry2, common2 = align_vectors(genes_ref, values_ref, genes_query2, values_query2)
    assert common2 == common
    assert np.allclose(vref2, vref)
    assert np.allclose(vqry2, vqry)


def test_standardize_vector_zero_variance_and_stats():
    x = np.array([5.0, 5.0, 5.0])
    z = standardize_vector(x)
    assert np.allclose(z, np.zeros_like(x))

    y = np.array([1.0, 2.0, 3.0])
    zy = standardize_vector(y)
    # Mean approximately 0 and std approximately 1
    assert abs(zy.mean()) < 1e-12
    assert abs(zy.std(ddof=0) - 1.0) < 1e-12

