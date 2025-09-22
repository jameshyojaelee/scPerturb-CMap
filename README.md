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
- Baseline: ensemble of cosine connectivity (lower is better) and GSEA-style enrichment (higher is better; flipped and z-scored to combine). Results now include per-signature z-scores and two-sided p-values derived from the blended ensemble.
- Metric: Dual-tower MLP (DualEncoder) trained with NT-Xent or Triplet loss on inversion pairs; scores are blended with the baseline. The training script can ingest real LINCS-derived inversion tables or fall back to synthetic toy data.

## Highlights
- **Replicate-aware preprocessing.** `scperturb-cmap score --collapse-replicates` performs MODZ-style collapse when a `replicate_id` column is available.
- **Richer target engineering.** `scperturb-cmap make-target` accepts pseudobulk grouping (`--pseudobulk-key`) and emits QC summaries (overlap counts, gene balance) that feed into the UI.
- **Pair generation for metric learning.** Utilities under `scperturb_cmap.data.pairs` build inversion pair tables with automatic negative sampling to bootstrap real-data training.
- **Dashboards.** The Streamlit UI surfaces target QC, MOA bar charts, and cell line heatmaps for quick storytelling.

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
  --collapse-replicates \
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

## Training on real inversion pairs
1. Build a table of known inversions with one row per `(target_id, signature_id)` pair (see `scperturb_cmap.data.pairs.prepare_pair_table`). Save as CSV/Parquet with columns `left_id`, `right_id`, `label` (1 for positives, 0 for negatives). If you only provide positives, the helper will sample negatives on the fly.
2. Export target signatures as JSON Lines where each record contains `target_id`, `genes`, and `weights` (e.g. produced by `scperturb-cmap make-target --qc-report`).
3. Run:

```bash
scperturb-cmap train pairs_path=/path/to/pairs.parquet \
  targets_path=/path/to/targets.jsonl \
  library_path=/path/to/lincs_long.parquet \
  negatives_per_target=5 \
  epochs=10 batch_size=128
```

The trainer auto-detects the gene dimension from the library subset, saves `artifacts/best.pt`, and logs `metrics.json` with the best validation recall.

### Preparing real LINCS L1000 Level 5

If you have Level 5 GCTX files + metadata, convert to long Parquet:

```bash
# Optional: derive the 978 landmark list from gene_info and save it
scperturb-cmap landmarks \
  --gene-info /path/to/gene_info.txt \
  --output data/l1000_landmarks.txt

# Convert GCTX → long and write a partitioned Parquet dataset by cell_line
scperturb-cmap prepare-lincs \
  --gctx /path/to/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ.gctx \
  --gene-info /path/to/gene_info.txt \
  --sig-info /path/to/GSE92742_Broad_LINCS_sig_info.txt.gz \
  --repurposing /path/to/repurposing_drugs.tsv \
  --landmarks \
  --partition-by cell_line \
  --output data/lincs/lincs_level5_landmark_long

# The output is a dataset directory; you can filter using --cell-line in score
scperturb-cmap score \
  --target-json examples/out/target.json \
  --library data/lincs/lincs_level5_landmark_long \
  --cell-lines A549 --cell-lines MCF7 \
  --moas "kinase inhibitor" --pert-types TRT_CP \
  --compounds TRAMETINIB \
  --dose-range 0.1,10 \
  --time-range 6,24 \
  --touchstone \
  --method baseline \
  --top-k 100 \
  --output examples/out/results_real.parquet
```

Notes:
- Provide `--landmarks-file` to point to a canonical 978 landmark list (else the tool looks for `data/l1000_landmarks.txt` or derives from `--gene-info`).
- You can also pass an existing long-form table via `--input` and filter to landmarks.
- For large libraries, prefer writing a partitioned dataset via `--partition-by cell_line` and use `--cell-line` when scoring to enable predicate pushdown.

## Devices: Mac vs HPC
- `--device auto` selects in order: `cuda` if available, else `mps` on Apple Silicon, else `cpu`.
- On MacBooks with Apple Silicon, PyTorch MPS is used automatically when available.
- On HPCs with NVIDIA GPUs, CUDA is used; otherwise CPU is used with reasonable defaults.

## Acceptance Criteria
1) Baseline scoring on the demo completes in under 60 seconds on a typical laptop and reports z-scores/p-values.
2) The DualEncoder improves Recall@50 over the baseline by at least 10% absolute on the demo (synthetic acceptance harness) when trained for a few epochs.
3) The UI loads the demo, surfaces target QC and MOA analytics, and can export a ranked CSV from the results table.

Run a basic acceptance check:

```bash
make acceptance
```

This script measures baseline scoring time, materializes a small inversion dataset under
`examples/out/metric_dataset/`, trains the DualEncoder with real pair/target files, and
verifies that recall@5 improves by at least 10 percentage points over the untrained model.

## Contributing

See CONTRIBUTING.md for development workflow and code style.

## License

MIT License. See `LICENSE` for details.
# The output is a dataset directory; you can filter using --cell-line in score
# scperturb-cmap score with multiple cell lines (repeat flag or list):
#   scperturb-cmap score --cell-line A549 --cell-line MCF7 ...
#
# Validate the dataset and show partition counts
python scripts/validate_parquet_dataset.py \
  --dataset data/lincs/lincs_level5_landmark_long \
  --partition cell_line \
  --full \
  --head 5
