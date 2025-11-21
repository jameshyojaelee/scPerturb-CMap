from __future__ import annotations

import pandas as pd
import pytest

from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.io.schemas import TargetSignature


def test_rank_drugs_raises_on_zero_overlap():
    target = TargetSignature(genes=["X1", "X2"], weights=[1.0, -1.0])
    library = pd.DataFrame(
        [
            {"signature_id": "sig1", "compound": "A", "cell_line": "CL", "gene_symbol": "G1", "score": 1.0},
            {"signature_id": "sig1", "compound": "A", "cell_line": "CL", "gene_symbol": "G2", "score": -1.0},
        ]
    )
    with pytest.raises(ValueError):
        rank_drugs(target, library, method="baseline", top_k=1)


def test_rank_drugs_emits_overlap_warning_metadata():
    genes = [f"G{i}" for i in range(10)]
    target = TargetSignature(genes=genes, weights=[1.0] * 10)
    library = pd.DataFrame(
        [
            {"signature_id": "sig1", "compound": "A", "cell_line": "CL", "gene_symbol": "G1", "score": 1.0},
            {"signature_id": "sig1", "compound": "A", "cell_line": "CL", "gene_symbol": "G2", "score": -1.0},
            {"signature_id": "sig2", "compound": "B", "cell_line": "CL", "gene_symbol": "G1", "score": -1.0},
            {"signature_id": "sig2", "compound": "B", "cell_line": "CL", "gene_symbol": "G2", "score": 1.0},
        ]
    )
    res = rank_drugs(target, library, method="baseline", top_k=2)
    assert res.metadata.get("overlap_warning") is True
    assert res.metadata.get("overlap_genes") == 2
