from __future__ import annotations

import numpy as np
import pandas as pd

from scperturb_cmap.data.lincs_gctx import GCTXData, gctx_to_long


def test_gctx_to_long_with_mocked_reader(tmp_path, monkeypatch):
    # Create a tiny synthetic matrix: genes x signatures
    genes = ["GAPDH", "ACTB", "TP53"]
    sigs = ["SIG_A", "SIG_B"]
    M = pd.DataFrame(
        data=np.array([[1.0, -1.0], [0.5, 0.0], [-2.0, 2.0]]),
        index=["rid1", "rid2", "rid3"],  # fake row ids
        columns=sigs,
    )
    # Row meta with gene symbols
    rmeta = pd.DataFrame({
        "pr_gene_symbol": genes,
    }, index=M.index)
    # Column meta with required fields
    cmeta = pd.DataFrame(
        {
            "sig_id": sigs,
            "pert_iname": ["DrugA", "DrugB"],
            "cell_id": ["A549", "A549"],
            "pert_type": ["TRT_CP", "TRT_CP"],
        }
    )

    # Monkeypatch read_gctx to return our synthetic object
    import scperturb_cmap.data.lincs_gctx as lg

    def _fake_read(_path: str) -> GCTXData:
        return GCTXData(matrix=M, row_meta=rmeta, col_meta=cmeta)

    monkeypatch.setattr(lg, "read_gctx", _fake_read)

    # Create a tiny repurposing TSV
    rep = pd.DataFrame({
        "pert_iname": ["DrugA", "DrugB"],
        "moa": ["MOA1", "MOA2"],
        "target": ["TG1", "TG2"],
    })
    rep_path = tmp_path / "rep.tsv"
    rep.to_csv(rep_path, sep="\t", index=False)

    # Run conversion with landmarks restricting to two genes
    out_path = tmp_path / "out.parquet"
    long_df = gctx_to_long(
        "dummy.gctx",
        gene_info_path=None,
        repurposing_path=str(rep_path),
        landmarks=["GAPDH", "TP53"],
        pert_type="TRT_CP",
        out_path=str(out_path),
    )

    # Either returns df or wrote to file; load if empty
    if long_df.empty and out_path.exists():
        long_df = pd.read_parquet(out_path, engine="pyarrow")

    # Check required columns
    req = {"signature_id", "compound", "cell_line", "gene_symbol", "score", "moa", "target"}
    assert req.issubset(set(long_df.columns))
    # Correct shapes: 2 genes * 2 signatures = 4 rows
    assert len(long_df) == 4
    # Landmark filter applied: no ACTB
    assert set(long_df["gene_symbol"]) == {"GAPDH", "TP53"}
    # Repurposing join applied
    assert set(long_df["moa"].dropna().unique()) == {"MOA1", "MOA2"}
