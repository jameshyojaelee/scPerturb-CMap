from __future__ import annotations

import numpy as np
import pandas as pd

from scperturb_cmap.io.schemas import TargetSignature
from scperturb_cmap.models.baseline import (
    cosine_connectivity,
    ensemble_connectivity,
    gsea_connectivity,
)


def make_target() -> TargetSignature:
    # Up: A, B; Down: C
    return TargetSignature(genes=["A", "B", "C"], weights=[1.0, 1.0, -1.0])


def test_cosine_connectivity_ordering():
    target = make_target()
    lincs_genes = ["A", "B", "C"]
    # pos aligned, neu weakly aligned, neg opposed
    M = np.array(
        [
            [2.0, 2.0, -2.0],
            [0.1, 0.0, -0.05],  # not proportional to pos
            [-2.0, -2.0, 2.0],
        ]
    )
    meta = pd.DataFrame(
        {
            "signature_id": ["pos", "neu", "neg"],
            "compound": ["P", "N", "N"],
            "cell_line": ["CL1", "CL1", "CL1"],
        }
    )

    df = cosine_connectivity(target, M, lincs_genes, meta)
    # Lower score (more negative) indicates stronger match
    scores = dict(zip(df["signature_id"], df["score"]))
    assert scores["pos"] < scores["neu"] < scores["neg"]


def test_gsea_connectivity_ordering():
    target = make_target()
    rows = []
    for sig, scores in (
        ("pos", {"A": 2.0, "B": 2.0, "C": -1.0}),
        ("neu", {"A": 0.1, "B": -0.05, "C": 0.0}),
        ("neg", {"A": -2.0, "B": -2.0, "C": 1.0}),
    ):
        for g, s in scores.items():
            rows.append(
                {
                    "signature_id": sig,
                    "compound": "P" if sig == "pos" else "N",
                    "cell_line": "CL1",
                    "gene_symbol": g,
                    "score": s,
                }
            )
    long_df = pd.DataFrame(rows)
    out = gsea_connectivity(target, long_df)
    s = dict(zip(out["signature_id"], out["score"]))
    assert s["pos"] > s["neu"] > s["neg"]


def test_ensemble_connectivity_combines_methods():
    target = make_target()
    lincs_genes = ["A", "B", "C"]
    M = np.array(
        [
            [2.0, 2.0, -2.0],
            [0.1, 0.0, -0.05],
            [-2.0, -2.0, 2.0],
        ]
    )
    meta = pd.DataFrame(
        {
            "signature_id": ["pos", "neu", "neg"],
            "compound": ["P", "N", "N"],
            "cell_line": ["CL1", "CL1", "CL1"],
        }
    )
    cos_df = cosine_connectivity(target, M, lincs_genes, meta)

    rows = []
    for sig, scores in (
        ("pos", {"A": 2.0, "B": 2.0, "C": -1.0}),
        ("neu", {"A": 0.1, "B": -0.05, "C": 0.0}),
        ("neg", {"A": -2.0, "B": -2.0, "C": 1.0}),
    ):
        for g, s in scores.items():
            rows.append(
                {
                    "signature_id": sig,
                    "compound": "P" if sig == "pos" else "N",
                    "cell_line": "CL1",
                    "gene_symbol": g,
                    "score": s,
                }
            )
    long_df = pd.DataFrame(rows)
    gsea_df = gsea_connectivity(target, long_df)

    ens = ensemble_connectivity(cos_df, gsea_df)
    e = dict(zip(ens["signature_id"], ens["score"]))
    # Lower is better after flipping GSEA inside ensemble
    assert e["pos"] < e["neu"] < e["neg"]
