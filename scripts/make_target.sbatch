#!/bin/bash
#SBATCH -J make_target
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o %x.%j.out
set -euo pipefail

echo "Starting target signature generation on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
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

DATA="/gpfs/commons/home/jameslee/scPerturb-CMap/data"
OUT="/gpfs/commons/home/jameslee/scPerturb-CMap/examples/out"
mkdir -p "$OUT"

# Default parameters (can be overridden by command line args)
H5AD="${1:-$DATA/sc/your_study.h5ad}"
CLUSTER_KEY="${2:-leiden}"
CLUSTER="${3:-2}"
REFERENCE="${4:-rest}"
OUTFILE="$OUT/target_sig_real.parquet"

echo "[make-target] Processing H5AD: $H5AD"
echo "  cluster_key=$CLUSTER_KEY"
echo "  cluster=$CLUSTER"
echo "  reference=$REFERENCE"
echo "  output=$OUTFILE"

# Generate target signature (CLI outputs JSON)
JSON_FILE="${OUTFILE%.parquet}.json"
scperturb-cmap make-target \
  --h5ad "$H5AD" \
  --cluster-key "$CLUSTER_KEY" \
  --cluster "$CLUSTER" \
  --reference "$REFERENCE" \
  --output "$JSON_FILE"

echo "[convert] Converting JSON to Parquet format..."
python - <<PY
import json
import pandas as pd
from pathlib import Path

json_file = Path("$JSON_FILE")
parquet_file = Path("$OUTFILE")

# Load JSON target signature
with open(json_file, 'r') as f:
    data = json.load(f)

# Convert to DataFrame format for Parquet
df = pd.DataFrame({
    'gene_symbol': data['genes'],
    'weight': data['weights']
})

# Add metadata as additional columns if present
if 'metadata' in data and data['metadata']:
    for key, value in data['metadata'].items():
        df[key] = value

# Save as Parquet
parquet_file.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(parquet_file, engine='pyarrow', index=False)

print(f"Converted {len(df)} genes to Parquet format")
PY

echo "[ok] Target signature generation and conversion complete"
echo "JSON output: $JSON_FILE"
echo "Parquet output: $OUTFILE"

# Show file info
if [[ -f "$OUTFILE" ]]; then
    ls -lh "$OUTFILE"
    echo "File size: $(stat -c%s "$OUTFILE") bytes"
else
    echo "Warning: Output file not found at $OUTFILE"
fi
