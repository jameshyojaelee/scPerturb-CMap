from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]


def load_parquet_table(path: PathLike) -> pd.DataFrame:
    """Load a Parquet table into a pandas DataFrame using pyarrow."""
    return pd.read_parquet(path, engine="pyarrow")


def save_parquet_table(df: pd.DataFrame, path: PathLike) -> None:
    """Save a pandas DataFrame to Parquet using pyarrow without the index."""
    pd.DataFrame(df).to_parquet(path, engine="pyarrow", index=False)

