#!/bin/bash
#SBATCH -J scpc_train
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o %x.%j.out
set -euo pipefail

echo "Starting metric learning training on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPUs: $SLURM_GPUS"
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "Allocated memory: $SLURM_MEM_PER_NODE"

# Activate conda environment if available
if command -v conda >/dev/null 2>&1; then
  echo "Activating conda environment..."
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate scpc || echo "Warning: scpc environment not found, continuing with system Python"
else
  echo "Conda not found, using system Python"
fi

PROJ="/gpfs/commons/home/jameslee/scPerturb-CMap"
LINCS="$PROJ/data/lincs/lincs_level5_landmark_long.parquet"
OUT="$PROJ/artifacts"

echo "Training configuration:"
echo "  LINCS data: $LINCS"
echo "  Output directory: $OUT"
echo "  Epochs: 20"
echo "  Batch size: 512"
echo "  Device: auto"

# Check if LINCS data exists
if [[ ! -f "$LINCS" ]]; then
    echo "Error: LINCS data not found at $LINCS"
    echo "Please run prepare_lincs.sbatch first"
    exit 1
fi

# Create output directory
mkdir -p "$OUT"

# Train the model with Hydra overrides
echo "Starting training..."
python -m scperturb_cmap.models.train \
  hydra.run.dir="$OUT" \
  epochs=20 \
  batch_size=512 \
  device=auto

echo "Training completed. Checking for best checkpoint..."

# Check if best.pt was created
BEST_CHECKPOINT="$OUT/best.pt"
if [[ ! -f "$BEST_CHECKPOINT" ]]; then
    echo "Error: Best checkpoint not found at $BEST_CHECKPOINT"
    echo "Training may have failed"
    exit 1
fi

echo "Evaluating trained model..."
REPORT_FILE="$OUT/report.json"

# Evaluate the trained model
scperturb-cmap evaluate --checkpoint "$BEST_CHECKPOINT" > "$REPORT_FILE"

echo "Evaluation completed. Results:"
python - <<PY
import json, os
p="$REPORT_FILE"
if os.path.exists(p):
    import json as J
    print('[ok] Training and evaluation metrics:')
    data = J.load(open(p))
    print(J.dumps(data, indent=2))
else:
    print('[err] missing report:', p)
PY

echo "[complete] Metric learning training finished"
