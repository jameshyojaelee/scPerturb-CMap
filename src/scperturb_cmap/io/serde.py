from __future__ import annotations

import pandas as pd


def load_parquet_table(path: str) -> pd.DataFrame:
    """Load a Parquet table into a pandas DataFrame using pyarrow."""
    return pd.read_parquet(path, engine="pyarrow")


def save_parquet_table(df: pd.DataFrame, path: str) -> None:
    """Save a pandas DataFrame to Parquet using pyarrow without the index."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")
    df.to_parquet(path, engine="pyarrow", index=False)
