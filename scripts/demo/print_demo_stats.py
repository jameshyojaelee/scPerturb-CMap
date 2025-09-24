#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scperturb_cmap.data.lincs_loader import load_lincs_long
from scperturb_cmap.data.scrna_loader import load_h5ad

EX_DIR = Path("examples/data")
LINCS_PATH = EX_DIR / "lincs_demo.parquet"
MECH_PATH = EX_DIR / "mechanisms.parquet"
H5AD_PATH = EX_DIR / "demo.h5ad"


def ensure_lincs_demo() -> Path:
    EX_DIR.mkdir(parents=True, exist_ok=True)
    if LINCS_PATH.exists() and MECH_PATH.exists():
        return LINCS_PATH

    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(1, 979)]
    compounds = [f"CMPD{i}" for i in range(1, 21)]
    cell_lines = ["CL1", "CL2", "CL3", "CL4", "CL5"]
    moa_classes = ["classA", "classB", "classC", "classD"]

    rows = []
    mechanisms = []
    for sid in range(200):
        compound = rng.choice(compounds)
        cell = rng.choice(cell_lines)
        moa = rng.choice(moa_classes)
        target = rng.choice(genes)
        mechanisms.append({"compound": compound, "moa": moa, "target": target})
        # signature_id as unique
        sig = f"sig{sid}"
        # Gene scores with a slight bias for the chosen target
        scores = rng.normal(loc=0.0, scale=1.0, size=len(genes))
        scores[genes.index(target)] += 2.5
        for g, s in zip(genes, scores):
            rows.append(
                {
                    "signature_id": sig,
                    "compound": compound,
                    "cell_line": cell,
                    "gene_symbol": g,
                    "score": float(s),
                }
            )
    df = pd.DataFrame(rows)
    mech = pd.DataFrame(mechanisms).drop_duplicates()
    df.to_parquet(LINCS_PATH, engine="pyarrow", index=False)
    mech.to_parquet(MECH_PATH, engine="pyarrow", index=False)
    return LINCS_PATH


def ensure_h5ad() -> Path:
    if H5AD_PATH.exists():
        return H5AD_PATH
    # Generate minimal synthetic AnnData without external imports
    import anndata as ad  # local import

    rng = np.random.default_rng(0)
    n_cells, n_genes, n_clusters = 5000, 978, 3
    genes = [f"G{i}" for i in range(1, n_genes + 1)]
    X = rng.normal(0.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
    clusters = np.array([f"C{i}" for i in range(n_clusters)])
    labels = rng.choice(clusters, size=n_cells, replace=True)
    # add signal
    for ci, cl in enumerate(clusters):
        idx = labels == cl
        start = ci * (n_genes // n_clusters)
        end = (ci + 1) * (n_genes // n_clusters)
        X[idx, start:end] += 2.0
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell{i}" for i in range(n_cells)]
    adata.var_names = genes
    adata.obs["cluster"] = labels
    EX_DIR.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(H5AD_PATH)
    return H5AD_PATH


def main() -> None:
    lincs_path = ensure_lincs_demo()
    h5ad_path = ensure_h5ad()

    df = load_lincs_long(str(lincs_path))
    n_rows = len(df)
    n_sigs = df["signature_id"].nunique()
    n_genes = df["gene_symbol"].nunique()
    print(f"LINCS long: rows={n_rows:,}, signatures={n_sigs:,}, genes={n_genes:,}")

    adata = load_h5ad(str(h5ad_path))
    n_cells = adata.n_obs
    n_vars = adata.n_vars
    n_clusters = adata.obs["cluster"].nunique()
    print(f"AnnData: cells={n_cells:,}, genes={n_vars:,}, clusters={n_clusters:,}")

    genes_lincs = set(df["gene_symbol"].astype(str))
    genes_adata = set(map(str, adata.var_names))
    overlap = len(genes_lincs & genes_adata)
    print(f"Gene overlap (LINCS vs .h5ad): {overlap} of {len(genes_lincs)}")

    if MECH_PATH.exists():
        mech = pd.read_parquet(MECH_PATH, engine="pyarrow")
        print(f"Mechanisms: {len(mech):,} rows; columns={list(mech.columns)}")


if __name__ == "__main__":
    main()
