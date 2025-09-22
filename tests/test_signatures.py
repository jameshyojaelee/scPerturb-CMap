from __future__ import annotations

import anndata as ad
import numpy as np

from scperturb_cmap.data.scrna_loader import load_h5ad
from scperturb_cmap.data.signatures import (
    summarize_target_signature,
    target_from_cells,
    target_from_cluster,
    target_from_gene_lists,
)


def make_toy_adata() -> ad.AnnData:
    # 4 cells, 3 genes
    X = np.array(
        [
            [10.0, 0.5, 1.0],  # T1 (cluster A)
            [11.0, 0.4, 1.0],  # T2 (cluster A)
            [1.0, 3.0, 1.0],   # R1 (cluster B)
            [2.0, 2.5, 1.0],   # R2 (cluster B)
        ]
    )
    obs = {
        "cluster": ["A", "A", "B", "B"],
        "patient": ["P1", "P1", "P1", "P2"],
    }
    var = {"gene_ids": ["g1", "g2", "g3"]}
    adata = ad.AnnData(X=X)
    adata.obs = adata.obs.assign(**obs)
    adata.var_names = ["G1", "G2", "G3"]
    adata.var = adata.var.assign(**var)
    adata.obs_names = ["T1", "T2", "R1", "R2"]
    return adata


def test_target_from_cluster_sign_direction(tmp_path):
    adata = make_toy_adata()
    ts = target_from_cluster(adata, cluster_key="cluster", cluster="A", reference="rest")
    genes = ts.genes
    weights = np.array(ts.weights)
    # Expect positive for G1 (up in A) and negative for G2 (down in A)
    idx_g1 = genes.index("G1")
    idx_g2 = genes.index("G2")
    idx_g3 = genes.index("G3")
    assert weights[idx_g1] > 0
    assert weights[idx_g2] < 0
    # G3 has no difference; should be near the middle
    assert abs(weights[idx_g3]) < max(abs(weights[idx_g1]), abs(weights[idx_g2]))


def test_target_from_cells_equivalence():
    adata = make_toy_adata()
    ts1 = target_from_cluster(adata, cluster_key="cluster", cluster="A")
    ts2 = target_from_cells(adata, target_barcodes=["T1", "T2"], ref_barcodes=["R1", "R2"])
    # Same genes, comparable signs
    assert ts1.genes == ts2.genes
    w1 = np.array(ts1.weights)
    w2 = np.array(ts2.weights)
    assert np.sign(w1).tolist() == np.sign(w2).tolist()


def test_target_from_gene_lists_standardization():
    ts = target_from_gene_lists(["A", "B"], ["C"])
    w = np.array(ts.weights)
    assert ts.genes == ["A", "B", "C"]
    # Standardized: roughly zero mean and unit std
    assert abs(w.mean()) < 1e-12
    assert abs(w.std(ddof=0) - 1.0) < 1e-12
    # Signs preserved
    assert w[0] > 0 and w[1] > 0 and w[2] < 0


def test_target_from_cluster_pseudobulk():
    adata = make_toy_adata()
    ts_bulk = target_from_cluster(
        adata,
        cluster_key="cluster",
        cluster="A",
        reference="rest",
        pseudobulk_key="patient",
    )
    assert len(ts_bulk.genes) == adata.n_vars


def test_summarize_target_signature_overlap():
    ts = target_from_gene_lists(["G1", "G2"], ["G3"])
    summary = summarize_target_signature(ts, library_genes=["G1", "G3", "G4"])
    assert summary["n_genes"] == 3
    assert summary["overlap_genes"] == 2
    assert summary["overlap_fraction"] == 2 / 3


def test_load_h5ad_roundtrip(tmp_path):
    adata = make_toy_adata()
    path = tmp_path / "toy.h5ad"
    adata.write_h5ad(path)
    loaded = load_h5ad(str(path))
    assert loaded.shape == adata.shape
    assert list(loaded.var_names) == list(adata.var_names)
