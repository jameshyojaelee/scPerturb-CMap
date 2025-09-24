from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import pandas as pd


def load_parquet_table(path: str) -> pd.DataFrame:
    """Load a Parquet table into a pandas DataFrame using pyarrow."""
    return pd.read_parquet(path, engine="pyarrow")


def save_parquet_table(df: pd.DataFrame, path: str) -> None:
    """Save a pandas DataFrame to Parquet using pyarrow without the index."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")
    df.to_parquet(path, engine="pyarrow", index=False)


def load_parquet_dataset_filtered(
    path: str,
    *,
    filter_expr=None,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load a Parquet dataset with an optional filter using pyarrow.dataset.

    This uses predicate pushdown where possible to minimize IO.
    ``filter_expr`` should be a pyarrow.dataset expression (e.g., ds.field('cell_line') == 'A549').
    """
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet")
    scanner = dataset.scanner(filter=filter_expr, columns=list(columns) if columns else None)
    table = scanner.to_table()
    return table.to_pandas()


def results_to_cellxgene(df: pd.DataFrame, *, obs_metadata: Dict[str, Any] | None = None):
    """Export ranked results into a minimal AnnData for cellxgene-like viewers.

    Creates a tiny variable space (score columns) and places rows as obs.
    """
    try:
        import anndata as ad  # type: ignore
        import numpy as np
    except Exception as e:  # pragma: no cover - optional
        raise RuntimeError("AnnData is required for cellxgene export") from e

    obs = df.copy()
    var_names = [c for c in ["score", "z_score", "p_value", "q_value"] if c in df.columns]
    var = pd.DataFrame(index=var_names)
    X = (
        np.zeros((len(obs), len(var.index)), dtype=float)
        if len(var.index)
        else np.zeros((len(obs), 0))
    )
    for j, col in enumerate(var.index):
        X[:, j] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
    if obs_metadata:
        for k, v in obs_metadata.items():
            obs[k] = v
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=var.index))
