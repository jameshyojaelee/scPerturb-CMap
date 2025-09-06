from __future__ import annotations

from typing import Iterable, Optional

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
