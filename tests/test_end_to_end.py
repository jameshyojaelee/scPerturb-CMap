from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.data.lincs_loader import load_lincs_long
from scperturb_cmap.io.schemas import TargetSignature
from scperturb_cmap.ui.helpers import (
    compute_contributions_from_library,
    load_target_signature_from_json_bytes,
    persist_exports,
)


def _ensure_demo_long() -> pd.DataFrame:
    demo_path = Path("examples/data/lincs_demo.parquet")
    if demo_path.exists():
        return load_lincs_long(str(demo_path))

    # Fallback: synthesize a tiny long-format table
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(1, 51)]
    rows = []
    for s in range(10):
        comp = f"C{s%3}"
        cl = f"CL{s%2}"
        for g in genes:
            rows.append(
                {
                    "signature_id": f"sig{s}",
                    "compound": comp,
                    "cell_line": cl,
                    "gene_symbol": g,
                    "score": float(rng.normal()),
                }
            )
    return pd.DataFrame(rows)


def test_end_to_end_baseline_scoring():
    # Build a small target signature
    ts = TargetSignature(genes=["G1", "G2", "G10"], weights=[1.0, 1.0, -1.0])
    df_long = _ensure_demo_long()

    res = rank_drugs(ts, df_long, method="baseline", top_k=10)
    # Convert to DataFrame for assertions
    ranking_df = (
        res.ranking if isinstance(res.ranking, pd.DataFrame) else pd.DataFrame(res.ranking)
    )

    # Must have required columns and nonempty rows
    required = {"signature_id", "compound", "cell_line", "score"}
    assert required.issubset(set(ranking_df.columns))
    assert len(ranking_df) > 0


def test_load_target_signature_from_json_bytes():
    payload = {"genes": ["G1", "G2"], "weights": [1.0, -1.0]}
    ts = load_target_signature_from_json_bytes(json.dumps(payload).encode())
    assert ts.genes == ["G1", "G2"]
    assert ts.weights[0] == 1.0


def test_compute_contributions_from_library():
    ts = TargetSignature(genes=["A", "B"], weights=[1.0, -1.0])
    library_df = pd.DataFrame(
        [
            {"signature_id": "sigA", "gene_symbol": "A", "score": -0.5},
            {"signature_id": "sigA", "gene_symbol": "B", "score": 0.25},
        ]
    )
    ranked = rank_drugs(ts, library_df, method="baseline", top_k=5)
    score_val = float(ranked.ranking.iloc[0]["score"])
    contrib = compute_contributions_from_library(
        ts,
        library_df,
        "sigA",
        final_score=score_val,
    )
    assert not contrib.empty
    assert "signature_id" in contrib.columns
    assert contrib["signature_id"].iloc[0] == "sigA"
    assert pytest.approx(contrib["contribution"].sum(), rel=1e-6) == score_val


def test_persist_exports(tmp_path: Path):
    csv_bytes = b"a,b\n1,2\n"
    json_bytes = b'{"x":1}'
    paths = persist_exports(tmp_path, "ui_test", csv_bytes, json_bytes, b"{}")
    assert paths["csv"].exists()
    assert paths["json"].exists()
    assert paths["session"].exists()
