from __future__ import annotations

import numpy as np
import pandas as pd

from scperturb_cmap.api import score as score_mod
from scperturb_cmap.data.lincs_loader import pivot_signatures
from scperturb_cmap.io.schemas import TargetSignature


def test_metric_scores_prefer_inversion(monkeypatch, tmp_path):
    def fake_load(_path, map_location=None):
        return {"config": {"input_dim": 3}, "state_dict": {}}

    class DummyModel:
        def __init__(self, input_dim: int, embed_dim: int = 64) -> None:
            self.input_dim = input_dim

        def load_state_dict(self, state):
            return

        def eval(self):
            return self

        def __call__(self, left, right):
            return left, right, (left * right).sum(dim=-1)

    monkeypatch.setattr(score_mod.torch, "load", fake_load)
    monkeypatch.setattr(score_mod, "DualEncoder", DummyModel)

    target = TargetSignature(genes=["G1", "G2", "G3"], weights=[1.0, 1.0, -1.0])
    library_df = pd.DataFrame(
        [
            {"signature_id": "sig_inverter", "compound": "A", "cell_line": "CL1", "gene_symbol": "G1", "score": -1.0},
            {"signature_id": "sig_inverter", "compound": "A", "cell_line": "CL1", "gene_symbol": "G2", "score": -1.0},
            {"signature_id": "sig_inverter", "compound": "A", "cell_line": "CL1", "gene_symbol": "G3", "score": 1.0},
            {"signature_id": "sig_concordant", "compound": "B", "cell_line": "CL1", "gene_symbol": "G1", "score": 1.0},
            {"signature_id": "sig_concordant", "compound": "B", "cell_line": "CL1", "gene_symbol": "G2", "score": 1.0},
            {"signature_id": "sig_concordant", "compound": "B", "cell_line": "CL1", "gene_symbol": "G3", "score": -1.0},
        ]
    )
    M, genes, meta = pivot_signatures(library_df)
    ckpt_path = tmp_path / "dummy.pt"
    ckpt_path.write_text("dummy")
    scores = score_mod._metric_scores(target, M, genes, str(ckpt_path))

    # Lower metric score should correspond to the inversion
    assert scores[0] < scores[1]
