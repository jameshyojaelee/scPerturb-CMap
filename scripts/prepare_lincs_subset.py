#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd

from scperturb_cmap.data.lincs_loader import load_lincs_long
from scperturb_cmap.data.preprocess import harmonize_symbols


def _read_gene_list(path: Path) -> List[str]:
    items: List[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        items.append(s)
    return harmonize_symbols(items)


def _demo_landmarks() -> List[str]:
    # Minimal built-in list for demo purposes only.
    # Replace with full L1000 landmark list as needed.
    return harmonize_symbols([
        "GAPDH",
        "ACTB",
        "TUBB",
        "RPLP0",
        "EEF1A1",
    ])


def filter_by_genes(df: pd.DataFrame, genes: Iterable[str]) -> pd.DataFrame:
    gene_set: Set[str] = set(harmonize_symbols(list(genes)))
    keep = df["gene_symbol"].astype(str).str.strip().str.upper().isin(gene_set)
    return df.loc[keep].reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a curated LINCS subset in long format and write Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to LINCS long file (csv/tsv/parquet)")
    parser.add_argument(
        "--output",
        default="examples/data/lincs_demo.parquet",
        help="Output Parquet path",
    )
    parser.add_argument(
        "--genes-file",
        help="Optional text file with one gene symbol per line; overrides --landmarks",
    )
    parser.add_argument(
        "--landmarks",
        action="store_true",
        help="Filter to a minimal built-in set of landmark-like genes (demo)",
    )
    args = parser.parse_args(argv)

    df = load_lincs_long(args.input)

    gene_list: List[str]
    if args.genes_file:
        gene_list = _read_gene_list(Path(args.genes_file))
    elif args.landmarks:
        gene_list = _demo_landmarks()
    else:
        gene_list = []

    if gene_list:
        df = filter_by_genes(df, gene_list)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)

    print(
        f"Wrote {len(df):,} rows, {df['signature_id'].nunique():,} signatures, "
        f"{df['gene_symbol'].nunique():,} genes -> {out_path}"
    )


if __name__ == "__main__":
    main()

