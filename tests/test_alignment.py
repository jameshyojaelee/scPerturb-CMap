from __future__ import annotations

import numpy as np

from scperturb_cmap.data.preprocess import align_vectors


def test_align_vectors_swapped_inputs_keep_identical_order_when_same_order():
    # Both inputs share identical order; swapping should keep the same order
    genes = ["A", "B", "C", "D"]
    v1 = np.array([1.0, 2.0, 3.0, 4.0])
    v2 = np.array([-1.0, -2.0, -3.0, -4.0])

    a1, b1, common1 = align_vectors(genes, v1, genes, v2)
    a2, b2, common2 = align_vectors(genes, v1, genes, v2)  # swapped would be same in this case

    assert common1 == common2
    assert np.allclose(a1, a2)
    assert np.allclose(b1, b2)

