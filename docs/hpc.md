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

Troubleshooting
---------------
- CPU-only nodes: the package runs on CPU; device detection falls back to CPU automatically.
- CUDA errors: ensure the PyTorch build matches your CUDA driver/toolkit on the cluster.
- Permissions: prefer user home or project scratch for `SCPC_BASE`.

