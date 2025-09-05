#!/bin/bash
#SBATCH -J prep_lincs
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o %x.%j.out
set -euo pipefail

echo "Starting LINCS preparation job on $(hostname)"
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

RAW="/gpfs/commons/home/jameslee/scPerturb-CMap/data/raw"
LNX="/gpfs/commons/home/jameslee/scPerturb-CMap/data/lincs"
mkdir -p "$LNX"

echo "Processing raw Level 5 files..."
echo "Input directory: $RAW"
echo "Output directory: $LNX"

# Process each Level 5 file
for f in "$RAW"/*Level5*.parquet "$RAW"/*Level5*.csv "$RAW"/*Level5*.tsv 2>/dev/null; do
  [[ -f "$f" ]] || continue
  base=$(basename "$f")
  out="$LNX/${base%.*}_long_landmark.parquet"
  echo "[prep] $f -> $out"
  python scripts/prepare_lincs_subset.py \
    --raw "$f" \
    --out "$out" \
    --landmark-only
done

echo "Merging processed files..."
python - <<'PY'
import glob, pandas as pd, os
root="/gpfs/commons/home/jameslee/scPerturb-CMap/data/lincs"
paths=glob.glob(os.path.join(root, '*_long_landmark.parquet'))
if paths:
    df=pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    out=os.path.join(root, 'lincs_level5_landmark_long.parquet')
    df.to_parquet(out, engine='pyarrow')
    print('[ok] merged ->', out, 'rows=', len(df))
    print(f'  signatures: {df["signature_id"].nunique():,}')
    print(f'  genes: {df["gene_symbol"].nunique():,}')
    print(f'  compounds: {df["compound"].nunique():,}')
else:
    print('[warn] no landmark parquet files found to merge')
PY

echo "[complete] LINCS preparation finished"
