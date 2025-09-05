#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def make_demo_h5ad(
    out_path: Path,
    n_cells: int = 5000,
    n_genes: int = 978,
    n_clusters: int = 3,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(1, n_genes + 1)]
    # Base expression
    X = rng.normal(loc=0.0, scale=1.0, size=(n_cells, n_genes)).astype(np.float32)
    # Cluster labels
    clusters = np.array([f"C{i}" for i in range(n_clusters)])
    labels = rng.choice(clusters, size=n_cells, replace=True)

    # Add differential signal across clusters
    for ci, cl in enumerate(clusters):
        idx = labels == cl
        # each cluster upregulates a disjoint block of genes
        start = ci * (n_genes // n_clusters)
        end = (ci + 1) * (n_genes // n_clusters)
        X[idx, start:end] += 2.0

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell{i}" for i in range(n_cells)]
    adata.var_names = genes
    adata.obs["cluster"] = pd.Categorical(labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic demo .h5ad")
    parser.add_argument(
        "--output",
        default="examples/data/demo.h5ad",
        help="Output .h5ad path",
    )
    parser.add_argument("--cells", type=int, default=5000)
    parser.add_argument("--genes", type=int, default=978)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = make_demo_h5ad(
        Path(args.output),
        n_cells=args.cells,
        n_genes=args.genes,
        n_clusters=args.clusters,
        seed=args.seed,
    )
    print(f"Wrote demo AnnData to {out}")


if __name__ == "__main__":
    main()
