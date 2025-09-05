#!/bin/bash
#SBATCH -J dl_lincs
#SBATCH -p cpu
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

# Ensure manifest has at least one URL
if ! grep -E '^[[:space:]]*https?://' -q lincs_level5_urls.txt; then
  echo "[err] No URLs found in lincs_level5_urls.txt. Please add real download URLs."
  exit 1
fi

if command -v aria2c >/dev/null 2>&1; then
  echo "Starting downloads with aria2c..."
  echo "Input file: lincs_level5_urls.txt"
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
  echo "[ok] Downloads complete (aria2c)"
  echo "Check aria2c_download.log for detailed output"
elif command -v wget >/dev/null 2>&1; then
  echo "aria2c not found; falling back to wget loop"
  while IFS= read -r url; do
    url="${url%%#*}"; url="${url//[$'\t\r\n ']}"; [[ -z "$url" ]] && continue
    echo "[wget] $url"
    wget --continue --tries=20 --timeout=30 "$url"
  done < lincs_level5_urls.txt
  echo "[ok] Downloads complete (wget)"
elif command -v curl >/dev/null 2>&1; then
  echo "aria2c/wget not found; falling back to curl loop"
  while IFS= read -r url; do
    url="${url%%#*}"; url="${url//[$'\t\r\n ']}"; [[ -z "$url" ]] && continue
    echo "[curl] $url"
    curl -L --retry 20 --retry-delay 10 -C - -O "$url"
  done < lincs_level5_urls.txt
  echo "[ok] Downloads complete (curl)"
else
  echo "[err] Neither aria2c, wget, nor curl found. Please install one of them."
  exit 1
fi

ls -lh *.gz *.txt *.parquet 2>/dev/null || echo "[warn] No downloaded files found in $RAW"
