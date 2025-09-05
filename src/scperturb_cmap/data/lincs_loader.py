from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLS = [
    "signature_id",
    "compound",
    "cell_line",
    "gene_symbol",
    "score",
]


def _ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_lincs_long(path: str) -> pd.DataFrame:
    """Load a curated LINCS table in long format.

    The table must contain the columns defined in REQUIRED_COLS.
    Supports CSV/TSV (by extension) and Parquet.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".parquet", ".parq"}:
        df = pd.read_parquet(p, engine="pyarrow")
    elif suffix in {".csv"}:
        df = pd.read_csv(p)
    elif suffix in {".tsv", ".txt"}:
        df = pd.read_csv(p, sep="\t")
    else:
        # Fallback: try parquet then CSV
        try:
            df = pd.read_parquet(p, engine="pyarrow")
        except Exception:
            df = pd.read_csv(p)

    _ensure_required_columns(df)

    # Basic type coercions
    df = df.copy()
    df["signature_id"] = df["signature_id"].astype(str)
    df["compound"] = df["compound"].astype(str)
    df["cell_line"] = df["cell_line"].astype(str)
    df["gene_symbol"] = df["gene_symbol"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    return df


def pivot_signatures(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Pivot long-form signatures to a matrix.

    Returns a tuple of:
    - S: ndarray of shape [num_signatures, num_genes]
    - genes: list of gene symbols used as columns
    - meta: DataFrame with columns ["signature_id", "compound", "cell_line"], one row per signature
    """
    _ensure_required_columns(df)
    # Metadata per signature in first-occurrence order
    meta = (
        df.loc[:, ["signature_id", "compound", "cell_line"]]
        .drop_duplicates(subset=["signature_id"], keep="first")
        .reset_index(drop=True)
    )

    # Deterministic gene order (sorted)
    genes = sorted(df["gene_symbol"].astype(str).unique().tolist())

    # Pivot with mean aggregator to handle potential duplicates
    wide = (
        pd.pivot_table(
            df,
            index="signature_id",
            columns="gene_symbol",
            values="score",
            aggfunc="mean",
        )
        .reindex(index=meta["signature_id"].tolist())
        .reindex(columns=genes)
    )

    S = wide.to_numpy(dtype=float)
    return S, genes, meta

