#!/bin/bash
#SBATCH -J dl_lincs
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -o %x.%j.out
set -euo pipefail

echo "Starting LINCS download job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

# Activate conda environment if available
if command -v conda >/dev/null 2>&1; then
  echo "Activating conda environment..."
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate scpc || echo "Warning: scpc environment not found, continuing with system Python"
else
  echo "Conda not found, using system Python"
fi

# Set working directory
RAW="/gpfs/commons/home/jameslee/scPerturb-CMap/data/raw"
cd "$RAW"
echo "Changed to directory: $(pwd)"

# Check for aria2c
if ! command -v aria2c >/dev/null 2>&1; then
  echo "Error: aria2c not found. Please ask admins to install aria2c or modify script to use curl/wget."
  echo "Installation command: conda install -c conda-forge aria2"
  exit 1
fi

echo "Starting downloads with aria2c..."
echo "Input file: lincs_level5_urls.txt"

# Run aria2c with robust download settings
aria2c \
  --input-file=lincs_level5_urls.txt \
  --continue=true \
  --max-connection-per-server=8 \
  --split=8 \
  --min-split-size=4M \
  --max-concurrent-downloads=4 \
  --retry-wait=10 \
  --max-tries=20 \
  --log-level=info \
  --log=aria2c_download.log

echo "[ok] Downloads complete"
echo "Check aria2c_download.log for detailed output"
ls -lh *.gz *.txt *.parquet 2>/dev/null || echo "No downloaded files found in current directory"
