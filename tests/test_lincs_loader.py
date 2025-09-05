from __future__ import annotations

import numpy as np
import pandas as pd

from scperturb_cmap.data.lincs_loader import load_lincs_long, pivot_signatures


def tiny_table() -> pd.DataFrame:
    rows = [
        {
            "signature_id": "sig1",
            "compound": "C1",
            "cell_line": "A549",
            "gene_symbol": "G1",
            "score": 1.0,
        },
        {
            "signature_id": "sig1",
            "compound": "C1",
            "cell_line": "A549",
            "gene_symbol": "G2",
            "score": 2.0,
        },
        {
            "signature_id": "sig2",
            "compound": "C2",
            "cell_line": "HEK293",
            "gene_symbol": "G2",
            "score": 3.0,
        },
        {
            "signature_id": "sig2",
            "compound": "C2",
            "cell_line": "HEK293",
            "gene_symbol": "G3",
            "score": 4.0,
        },
    ]
    return pd.DataFrame(rows)


def test_load_lincs_long_from_csv_and_parquet(tmp_path):
    df = tiny_table()
    csv_path = tmp_path / "tiny.csv"
    pq_path = tmp_path / "tiny.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, engine="pyarrow", index=False)

    df_csv = load_lincs_long(str(csv_path))
    df_pq = load_lincs_long(str(pq_path))

    assert set(df_csv.columns) >= {
        "signature_id",
        "compound",
        "cell_line",
        "gene_symbol",
        "score",
    }
    assert df_csv.shape == df.shape
    assert df_pq.shape == df.shape


def test_pivot_signatures_shapes_and_values():
    df = tiny_table()
    S, genes, meta = pivot_signatures(df)
    assert genes == ["G1", "G2", "G3"]
    assert list(meta.columns) == ["signature_id", "compound", "cell_line"]
    assert meta.shape[0] == 2
    assert S.shape == (2, 3)
    row0 = S[0]
    row1 = S[1]
    # sig1: G1=1, G2=2, G3 missing
    assert np.allclose(row0[:2], np.array([1.0, 2.0]))
    assert np.isnan(row0[2])
    # sig2: G1 missing, G2=3, G3=4
    assert np.isnan(row1[0])
    assert np.allclose(row1[1:], np.array([3.0, 4.0]))
