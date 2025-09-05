from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.data.lincs_loader import load_lincs_long
from scperturb_cmap.io.schemas import TargetSignature


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

