HPC Setup
=========

This guide covers running scPerturb-CMap on an HPC cluster. It keeps site-specific paths configurable while leaving the codebase portable.

Prerequisites
-------------
- Python 3.10 (via Conda or system Python)
- Optional: `mamba` (faster environment solving)
- PyTorch CPU or CUDA build (CUDA if using NVIDIA GPUs)

Directory Layout
----------------
You can keep data and outputs outside the repo by setting `SCPC_BASE`. The setup creates:

- `data/raw/` – raw downloads
- `data/lincs/` – LINCS tables (parquet/csv)
- `data/sc/` – single-cell inputs (e.g., `.h5ad`)
- `artifacts/` – model checkpoints and metrics
- `examples/out/` – quickstart outputs

Setup
-----
From the repository root:

```bash
# Optional: choose where to place data and outputs
export SCPC_BASE=/gpfs/commons/home/$USER/scPerturb-CMap

# Create env and install package + dev deps
make hpc-setup

# Verify device detection
.venv/bin/python -c "from scperturb_cmap.utils.device import get_device; print(get_device())"
```

Notes on Environments
---------------------
- The `Makefile` uses a local `.venv/` for development. The `scripts/setup_hpc.sh` script will prefer Conda/Mamba if found and create/use an env named `scpc`.
- If your site requires modules (e.g., `module load cuda/12.1`), load them before `make hpc-setup`.

Running Jobs (Slurm example)
----------------------------
Minimal template to run scoring; adapt partition, time, and GPUs as needed:

```bash
#!/bin/bash
#SBATCH -J scpc-score
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o slurm-%x-%j.out

set -euo pipefail
module purge
# module load cuda/12.1  # if your site requires it

export SCPC_BASE=/gpfs/commons/home/$USER/scPerturb-CMap
cd $SLURM_SUBMIT_DIR

# Activate your environment (choose one)
# 1) Conda/Mamba environment created by setup_hpc.sh
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate scpc
fi

# 2) Or local venv if you used `make setup`
# source .venv/bin/activate

# Run a small demo
python -m scperturb_cmap.cli score \
  --target-json examples/out/target.json \
  --library examples/data/lincs_demo.parquet \
  --method baseline \
  --top-k 50 \
  --output ${SCPC_BASE}/examples/out/results.parquet
```

Data Ingestion
--------------
- Place LINCS long table(s) under `${SCPC_BASE}/data/lincs/`.
- Place your `.h5ad` single-cell inputs under `${SCPC_BASE}/data/sc/`.
- Update your paths or pass them as CLI args via `--library` and your target JSON.

### Converting GCTX Level 5 to long Parquet

If you download Level 5 GCTX files, you can use either the CLI directly or the Slurm job to convert them into a long table with annotations.

Option A — Run CLI directly (interactive node):

```bash
# Optional: extract landmark genes from gene_info
scperturb-cmap landmarks --gene-info $RAW/gene_info.txt --output $SCPC_BASE/data/l1000_landmarks.txt

# Convert GCTX and write a partitioned dataset by cell_line (recommended for scale)
scperturb-cmap prepare-lincs \
  --gctx $RAW/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ.gctx \
  --gene-info $RAW/gene_info.txt \
  --sig-info $RAW/GSE92742_Broad_LINCS_sig_info.txt.gz \
  --repurposing $RAW/repurposing_drugs.tsv \
  --landmarks \
  --partition-by cell_line \
  --output $SCPC_BASE/data/lincs/lincs_level5_landmark_long
```

The job writes into a single dataset directory:
`$SCPC_BASE/data/lincs/lincs_level5_landmark_long/`.
Use CLI `score --cell-line <ID>` to leverage predicate pushdown when reading.

Option B — Submit a Slurm job (recommended on headless clusters):

```bash
# Ensure you have downloaded raw files into $SCPC_BASE/data/raw first.
# Then submit the conversion job:
sbatch scripts/convert_lincs_gctx.sbatch

# Optional environment overrides (set before sbatch):
#   SCPC_BASE=/path/to/base         # default: repo root
#   GCTX=/path/to/file.gctx         # default: auto-detect in $SCPC_BASE/data/raw
#   GENE_INFO=/path/to/gene_info.gz # default: auto-detect
#   SIG_INFO=/path/to/sig_info.gz   # default: auto-detect
#   REPURPOSING=/path/to/rep.tsv    # optional Repurposing Hub annotations
#   CHUNK_COLS=2000                 # columns per chunk during write
#   PARTITION=cell_line             # partition column (default)
```

The job writes a partitioned Parquet dataset by `cell_line` under
`$SCPC_BASE/data/lincs/lincs_level5_landmark_long/` and validates it at the end.

Troubleshooting
---------------
- CPU-only nodes: the package runs on CPU; device detection falls back to CPU automatically.
- CUDA errors: ensure the PyTorch build matches your CUDA driver/toolkit on the cluster.
- Permissions: prefer user home or project scratch for `SCPC_BASE`.
