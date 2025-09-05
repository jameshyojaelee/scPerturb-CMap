from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.io.schemas import TargetSignature
from scperturb_cmap.utils.seed import set_global_seed


def test_min_overlap_guardrail_raises():
    # Library with genes that don't overlap the target
    genes_lib = [f"X{i}" for i in range(1, 1000)]
    # Build long-form minimal DataFrame for pivot to work via API
    import pandas as pd

    rows = []
    for s in range(5):
        for g in genes_lib:
            rows.append({
                "signature_id": f"sig{s}",
                "compound": f"C{s}",
                "cell_line": "CL",
                "gene_symbol": g,
                "score": float(s),
            })
    lib_df = pd.DataFrame(rows)

    # Target with >=300 genes to trigger guardrail
    target_genes = [f"G{i}" for i in range(1, 400)]
    ts = TargetSignature(genes=target_genes, weights=[1.0] * len(target_genes))

    with pytest.raises(ValueError) as ei:
        rank_drugs(ts, lib_df, method="baseline", top_k=5)
    # Helpful error lists some missing genes
    assert "Insufficient gene overlap" in str(ei.value)
    assert "G1" in str(ei.value)


def test_set_global_seed_deterministic():
    set_global_seed(123)
    _ = np.random.RandomState(0).randn(3)  # unrelated RNG
    x1 = np.random.randn(3)
    t1 = torch.randn(3)
    r1 = random.random()

    set_global_seed(123)
    _ = np.random.RandomState(0).randn(3)
    x2 = np.random.randn(3)
    t2 = torch.randn(3)
    r2 = random.random()

    assert np.allclose(x1, x2)
    assert torch.allclose(t1, t2)
    assert r1 == r2
