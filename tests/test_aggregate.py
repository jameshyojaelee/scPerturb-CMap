from __future__ import annotations

import pandas as pd

from scperturb_cmap.analysis.aggregate import collapse_replicates_modz


def test_collapse_replicates_modz_weights_high_similarity():
    rows = []
    for rep, vec in (
        ("r1", [1.0, 2.0, -1.0]),
        ("r2", [1.1, 2.1, -1.2]),
        ("r3", [-0.5, -1.0, 0.3]),
    ):
        for gene, val in zip(["A", "B", "C"], vec):
            rows.append(
                {
                    "signature_id": "sigA",
                    "replicate_id": rep,
                    "gene_symbol": gene,
                    "score": val,
                    "compound": "CMPD",
                    "cell_line": "CL",
                }
            )
    df = pd.DataFrame(rows)

    collapsed = collapse_replicates_modz(df)
    assert {"signature_id", "gene_symbol", "score"}.issubset(set(collapsed.columns))
    assert collapsed["signature_id"].nunique() == 1
    assert set(collapsed["gene_symbol"]) == {"A", "B", "C"}

    scores = dict(zip(collapsed["gene_symbol"], collapsed["score"]))
    # Expect genes A/B to remain positive, C negative after weighting
    assert scores["A"] > 0
    assert scores["B"] > 0
    assert scores["C"] < 0


def test_collapse_replicates_modz_missing_replicate_column_returns_copy():
    df = pd.DataFrame(
        {
            "signature_id": ["sig1", "sig1"],
            "gene_symbol": ["A", "B"],
            "score": [0.1, -0.1],
        }
    )
    collapsed = collapse_replicates_modz(df)
    pd.testing.assert_frame_equal(df, collapsed)
