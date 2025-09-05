from __future__ import annotations

import anndata as ad


def load_h5ad(path: str) -> ad.AnnData:
    """Load an AnnData object from an .h5ad file path."""
    return ad.read_h5ad(path)

