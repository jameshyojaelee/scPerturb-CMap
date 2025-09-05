# scPerturb-CMap

Single-cell connectivity mapping with baseline and deep metric learning for precision drug repurposing.

## Problem Statement

We want to find drugs that invert a disease signature observed in a specific cell population (e.g., a resistant tumor subclone). Traditional bulk connectivity mapping averages across cells and can miss rare, clinically relevant states. scPerturb-CMap focusses on single cells or clusters and supports both a strong baseline and a learnable (metric) approach.

## Data Flow

```
  scRNA-seq (.h5ad)  ──► target signature (genes, weights)
                              │
                              ▼
  LINCS long (parquet) ──► align + score (baseline | metric) ──► ranked drugs
```

## Data Formats
- Target signature (JSON via Pydantic): `{ "genes": [str], "weights": [float], "metadata": {...} }`
- LINCS long (parquet/csv): columns `signature_id, compound, cell_line, gene_symbol, score` (+ optional `moa, target`)
- Results (parquet/csv): `signature_id, compound, cell_line, moa?, target?, score`

## Methods
- Baseline: ensemble of cosine connectivity (lower is better) and GSEA-style enrichment (higher is better; flipped and z-scored to combine). No training required.
- Metric: Dual-tower MLP (DualEncoder) trained with NT-Xent or Triplet loss on inversion pairs; scores are blended with the baseline.

## Quickstart

```bash
# Setup environment and install
make setup

# Generate synthetic demo data and print stats
make demo

# Run baseline scoring on demo (writes examples/out/results.parquet)
scperturb-cmap score \
  --target-json examples/out/target.json \
  --library examples/data/lincs_demo.parquet \
  --method baseline --top-k 50 --output examples/out/results.parquet

# Launch the Streamlit demo UI
make ui

# Train a tiny model on synthetic pairs and evaluate
make train
make evaluate

# Lint and test
make lint
make test
```

## HPC Setup
- For cluster-specific setup, directories, and Slurm examples, see `docs/hpc.md`.

## Devices: Mac vs HPC
- `--device auto` selects in order: `cuda` if available, else `mps` on Apple Silicon, else `cpu`.
- On MacBooks with Apple Silicon, PyTorch MPS is used automatically when available.
- On HPCs with NVIDIA GPUs, CUDA is used; otherwise CPU is used with reasonable defaults.

## Acceptance Criteria
1) Baseline scoring on the demo completes in under 60 seconds on a typical laptop.
2) The DualEncoder improves Recall@50 over the baseline by at least 10% absolute on the demo (synthetic acceptance harness) when trained for a few epochs.
3) The UI loads the demo and can export a ranked CSV from the results table.

Run a basic acceptance check:

```bash
make acceptance
```

This script measures baseline scoring time, trains a short model, and verifies recall improvement on a synthetic retrieval task.

## Contributing

See CONTRIBUTING.md for development workflow and code style.

## License

MIT License. See `LICENSE` for details.
