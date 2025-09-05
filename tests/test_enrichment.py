from __future__ import annotations

import pandas as pd

from scperturb_cmap.analysis.enrichment import moa_enrichment


def test_moa_enrichment_runs():
    df = pd.DataFrame(
        {
            "compound": ["A", "B", "C", "D"],
            "moa": ["X", "Y", "X", "Y"],
            "score": [-2.0, -1.0, 0.5, 1.0],
        }
    )
    out = moa_enrichment(df, top_n=2)
    assert {"moa", "top", "rest", "odds_ratio", "p_value"}.issubset(out.columns)

