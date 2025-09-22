#!/bin/bash
#SBATCH -J prep_lincs
#SBATCH -p cpu
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

PROJ="/gpfs/commons/home/jameslee/scPerturb-CMap"
RAW="$PROJ/data/raw"
LNX="$PROJ/data/lincs"
mkdir -p "$LNX"

echo "Processing raw Level 5 files..."
echo "Input directory: $RAW"
echo "Output directory: $LNX"

files_found=0
shopt -s nullglob

# Prefer GCTX inputs
for g in "$RAW"/*.gctx; do
  [[ -f "$g" ]] || continue
  base=$(basename "$g")
  out="$LNX/lincs_level5_landmark_long"  # partitioned dataset directory
  echo "[prep] GCTX $g -> $out (partitioned by cell_line)"
  # Optional metadata files if present
  GENE_INFO=""; REPURPOSING=""; SIG_INFO=""; INST_INFO=""
  for c in "$RAW"/gene_info*.txt*; do [[ -f "$c" ]] && GENE_INFO="$c" && break; done
  for r in "$RAW"/repurposing*.*; do [[ -f "$r" ]] && REPURPOSING="$r" && break; done
  for s in "$RAW"/GSE*_sig_info*.txt*; do [[ -f "$s" ]] && SIG_INFO="$s" && break; done
  for i in "$RAW"/GSE*_inst_info*.txt*; do [[ -f "$i" ]] && INST_INFO="$i" && break; done
  scperturb-cmap prepare-lincs \
    --gctx "$g" \
    ${GENE_INFO:+--gene-info "$GENE_INFO"} \
    ${REPURPOSING:+--repurposing "$REPURPOSING"} \
    ${SIG_INFO:+--sig-info "$SIG_INFO"} \
    ${INST_INFO:+--inst-info "$INST_INFO"} \
    --landmarks \
    --partition-by cell_line \
    --output "$out"
  ((files_found++))
done

# Fall back to any long-form CSV/Parquet/TSV in RAW
if [[ $files_found -eq 0 ]]; then
  for f in "$RAW"/*Level5*.parquet "$RAW"/*Level5*.csv "$RAW"/*Level5*.tsv; do
    [[ -f "$f" ]] || continue
    base=$(basename "$f")
    out="$LNX/${base%.*}_long_landmark.parquet"
    echo "[prep] long $f -> $out"
    scperturb-cmap prepare-lincs \
      --input "$f" \
      --landmarks \
      --output "$out"
    ((files_found++))
  done
fi

# If still nothing, use demo data
if [[ $files_found -eq 0 ]]; then
  echo "[info] No downloaded files found, using demo data"
  demo_file="$PROJ/examples/data/lincs_demo.parquet"
  if [[ -f "$demo_file" ]]; then
    out="$LNX/lincs_demo_long_landmark.parquet"
    echo "[prep] demo $demo_file -> $out"
    scperturb-cmap prepare-lincs --input "$demo_file" --landmarks --output "$out"
    ((files_found++))
  else
    echo "[err] No downloaded or demo files found"
    exit 1
  fi
fi

echo "[complete] LINCS preparation finished (dataset: $LNX/lincs_level5_landmark_long)"
