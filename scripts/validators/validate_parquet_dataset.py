#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def _count_rows_dataset(path: str) -> int:
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet")
    return int(dataset.count_rows())


def _partition_counts(path: str, partition_col: str) -> pd.Series:
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet")
    # Scan only the partition column to minimize IO
    table = dataset.scanner(columns=[partition_col]).to_table()
    df = table.to_pandas()
    if partition_col not in df.columns:
        return pd.Series(dtype=int)
    return df[partition_col].astype(str).value_counts().sort_index()


def _basic_stats(path: str) -> dict:
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet")
    cols = [c for c in ["signature_id", "gene_symbol", "compound"] if c in dataset.schema.names]
    if not cols:
        return {}
    table = dataset.scanner(columns=cols).to_table()
    df = table.to_pandas()
    out = {}
    if "signature_id" in df:
        out["signatures"] = int(df["signature_id"].nunique())
    if "gene_symbol" in df:
        out["genes"] = int(df["gene_symbol"].nunique())
    if "compound" in df:
        out["compounds"] = int(df["compound"].nunique())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a Parquet dataset and print partition counts.")
    ap.add_argument("--dataset", required=True, help="Path to a Parquet file or dataset directory")
    ap.add_argument("--partition", default="cell_line", help="Partition column to summarize (default: cell_line)")
    ap.add_argument("--full", action="store_true", help="Compute additional unique counts (signatures/genes/compounds)")
    ap.add_argument("--head", type=int, default=0, help="Optionally print the first N rows of the dataset")
    args = ap.parse_args()

    p = Path(args.dataset)
    if not p.exists():
        raise SystemExit(f"Dataset not found: {p}")

    try:
        n_rows = _count_rows_dataset(str(p))
    except Exception as e:
        raise SystemExit(f"Failed to read dataset: {e}")

    print(f"Dataset: {p}")
    print(f"Rows: {n_rows:,}")

    try:
        vc = _partition_counts(str(p), args.partition)
        if vc.empty:
            print(f"Partition column '{args.partition}' not found or no values")
        else:
            print(f"Partitions by '{args.partition}': {len(vc)} values")
            for k, v in vc.items():
                print(f"  {k}: {v:,}")
    except Exception as e:
        print(f"[warn] Failed to compute partition counts: {e}")

    if args.full:
        try:
            stats = _basic_stats(str(p))
            if stats:
                print("Basic stats:")
                for k in ["signatures", "genes", "compounds"]:
                    if k in stats:
                        print(f"  {k}: {stats[k]:,}")
        except Exception as e:
            print(f"[warn] Failed to compute basic stats: {e}")

    if args.head > 0:
        try:
            import pyarrow.dataset as ds

            dataset = ds.dataset(str(p), format="parquet")
            # Load only a small sample of columns for the head
            cols = [c for c in ["signature_id", "compound", "cell_line", "gene_symbol", "score"] if c in dataset.schema.names]
            df = dataset.scanner(columns=cols).to_table().to_pandas()
            print(df.head(args.head).to_string(index=False))
        except Exception as e:
            print(f"[warn] Failed to print head: {e}")


if __name__ == "__main__":
    main()

