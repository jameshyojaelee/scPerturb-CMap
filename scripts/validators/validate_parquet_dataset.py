#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from scperturb_cmap.data.lincs_loader import REQUIRED_COLS


def _partition_counts(dataset, partition_col: str) -> pd.Series:
    # Scan only the partition column to minimize IO
    table = dataset.scanner(columns=[partition_col]).to_table()
    df = table.to_pandas()
    if partition_col not in df.columns:
        return pd.Series(dtype=int)
    return df[partition_col].astype(str).value_counts().sort_index()


def _basic_stats(dataset) -> dict:
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


def _schema_report(dataset, sample_size: int = 5000) -> Dict[str, Dict[str, float]]:
    schema_names = set(dataset.schema.names)
    missing = sorted(set(REQUIRED_COLS) - schema_names)
    extras = sorted(schema_names - set(REQUIRED_COLS))
    dtypes = {field.name: str(field.type) for field in dataset.schema}

    null_fraction: Dict[str, float] = {}
    if sample_size > 0:
        try:
            table = dataset.head(sample_size)
        except Exception:
            table = dataset.scanner(limit=sample_size).to_table()
        df = table.to_pandas()
        for col in df.columns:
            null_fraction[col] = float(df[col].isna().mean())

    return {
        "missing_columns": missing,
        "extra_columns": extras,
        "dtypes": dtypes,
        "null_fraction_sample": null_fraction,
    }


def _head(dataset, cols, limit: int) -> Optional[pd.DataFrame]:
    if limit <= 0:
        return None
    columns = [c for c in cols if c in dataset.schema.names]
    if not columns:
        return None
    table = dataset.scanner(columns=columns, limit=limit).to_table()
    return table.to_pandas()


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a Parquet dataset and print partition counts.")
    ap.add_argument("--dataset", required=True, help="Path to a Parquet file or dataset directory")
    ap.add_argument("--partition", default="cell_line", help="Partition column to summarize (default: cell_line)")
    ap.add_argument("--full", action="store_true", help="Compute additional unique counts (signatures/genes/compounds)")
    ap.add_argument("--head", type=int, default=0, help="Optionally print the first N rows of the dataset")
    ap.add_argument("--sample-size", type=int, default=5000, help="Rows to sample for null-fraction estimation")
    ap.add_argument("--metrics-json", type=str, default=None, help="Optional path to write metrics as JSON")
    args = ap.parse_args()

    p = Path(args.dataset)
    if not p.exists():
        raise SystemExit(f"Dataset not found: {p}")

    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise SystemExit("pyarrow is required to validate Parquet datasets") from exc

    try:
        dataset = ds.dataset(str(p), format="parquet")
        n_rows = int(dataset.count_rows())
    except Exception as e:
        raise SystemExit(f"Failed to read dataset: {e}")

    print(f"Dataset: {p}")
    print(f"Rows: {n_rows:,}")

    metrics: Dict[str, object] = {"rows": n_rows}

    try:
        vc = _partition_counts(dataset, args.partition)
        if vc.empty:
            print(f"Partition column '{args.partition}' not found or no values")
            metrics["partitions"] = {}
        else:
            print(f"Partitions by '{args.partition}': {len(vc)} values")
            for k, v in vc.items():
                print(f"  {k}: {v:,}")
            metrics["partitions"] = vc.to_dict()
    except Exception as e:
        print(f"[warn] Failed to compute partition counts: {e}")

    if args.full:
        try:
            stats = _basic_stats(dataset)
            metrics["uniques"] = stats
            if stats:
                print("Basic stats:")
                for k in ["signatures", "genes", "compounds"]:
                    if k in stats:
                        print(f"  {k}: {stats[k]:,}")
        except Exception as e:
            print(f"[warn] Failed to compute basic stats: {e}")

    try:
        schema_report = _schema_report(dataset, sample_size=args.sample_size)
        metrics["schema"] = schema_report
        if schema_report["missing_columns"] or schema_report["extra_columns"]:
            print("Schema drift:")
            if schema_report["missing_columns"]:
                print(f"  Missing: {', '.join(schema_report['missing_columns'])}")
            if schema_report["extra_columns"]:
                print(f"  Extras: {', '.join(schema_report['extra_columns'])}")
        else:
            print("Schema aligns with expected LINCS columns.")
    except Exception as e:
        print(f"[warn] Failed to compute schema report: {e}")

    head_df = _head(
        dataset,
        cols=["signature_id", "compound", "cell_line", "gene_symbol", "score"],
        limit=args.head,
    )
    if head_df is not None:
        print(head_df.head(args.head).to_string(index=False))

    if args.metrics_json:
        out_path = Path(args.metrics_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"Wrote metrics -> {out_path}")


if __name__ == "__main__":
    main()
