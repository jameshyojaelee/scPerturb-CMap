from __future__ import annotations

from pathlib import Path

import pandas as pd

from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.io.schemas import TargetSignature


def test_rank_drugs_respects_cell_line_filters(tmp_path: Path):
    data = pd.DataFrame(
        [
            {"signature_id": "sig_a", "compound": "A", "cell_line": "A", "gene_symbol": "G1", "score": -1.0},
            {"signature_id": "sig_a", "compound": "A", "cell_line": "A", "gene_symbol": "G2", "score": 1.0},
            {"signature_id": "sig_b", "compound": "B", "cell_line": "B", "gene_symbol": "G1", "score": 1.0},
            {"signature_id": "sig_b", "compound": "B", "cell_line": "B", "gene_symbol": "G2", "score": -1.0},
        ]
    )
    dataset_dir = tmp_path / "dataset"
    data.to_parquet(dataset_dir, engine="pyarrow", partition_cols=["cell_line"])

    target = TargetSignature(genes=["G1", "G2"], weights=[1.0, -1.0])
    result = rank_drugs(
        target_signature=target,
        library=dataset_dir,
        method="baseline",
        top_k=5,
        filters={"cell_line": ["A"]},
    )
    ranking_df = result.ranking if isinstance(result.ranking, pd.DataFrame) else pd.DataFrame(result.ranking)
    assert set(ranking_df["cell_line"].unique()) == {"A"}
