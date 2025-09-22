#!/bin/bash
# Convert LINCS Level 5 GCTX to landmark-only long Parquet using sig_info metadata
# Writes a partitioned dataset by cell_line for efficient filtering.
#
# Usage:
#   sbatch scripts/convert_lincs_gctx.sbatch
#
# Optional env overrides:
#   SCPC_BASE=/path/to/base         # base dir for data (defaults to repo root)
#   GCTX=/path/to/file.gctx         # explicit GCTX path (else auto-detect in $SCPC_BASE/data/raw)
#   GENE_INFO=/path/to/gene_info.gz # explicit gene_info path (auto-detect if unset)
#   SIG_INFO=/path/to/sig_info.gz   # explicit sig_info path (auto-detect if unset)
#   REPURPOSING=/path/to/rep.tsv    # optional Repurposing Hub annotations
#   CHUNK_COLS=2000                 # columns per chunk during write (memory control)
#   PARTITION=cell_line             # partition column (default: cell_line)

#SBATCH -J convert_lincs
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o %x.%j.out

set -euo pipefail

# Resolve submission directory (original repo) and project base
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SUBMIT_DIR"
repo_root="$SUBMIT_DIR"
PROJ="${SCPC_BASE:-$SUBMIT_DIR}"
RAW="$PROJ/data/raw"
OUT_DIR="$PROJ/data/lincs/lincs_level5_landmark_long"
CHUNK_COLS="${CHUNK_COLS:-2000}"
PARTITION_COL="${PARTITION:-cell_line}"

mkdir -p "$PROJ/data/lincs"

echo "[info] PROJ=$PROJ"
echo "[info] RAW=$RAW"

echo "[info] Activating environment..."
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate scpc || echo "[warn] conda env 'scpc' not found; using system Python"
elif [[ -f "$repo_root/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$repo_root/.venv/bin/activate"
fi

# Detect inputs if not provided
if [[ -z "${GCTX:-}" ]]; then
  # Prefer uncompressed GCTX; else decompress the .gz
  cand=$(ls -1 "$RAW"/*Level5*.gctx 2>/dev/null | head -n 1 || true)
  if [[ -z "$cand" ]]; then
    gz=$(ls -1 "$RAW"/*Level5*.gctx.gz 2>/dev/null | head -n 1 || true)
    if [[ -n "$gz" ]]; then
      echo "[info] Decompressing $gz ..."
      GCTX="$RAW/$(basename "${gz%.gz}")"
      gunzip -c "$gz" > "$GCTX"
    fi
  else
    GCTX="$cand"
  fi
fi

# Auto-detect metadata
GENE_INFO="${GENE_INFO:-}"
if [[ -z "$GENE_INFO" ]]; then
  for f in "$RAW"/GSE*_gene_info*.txt* "$RAW"/gene_info*.txt*; do
    [[ -f "$f" ]] && GENE_INFO="$f" && break
  done
fi
SIG_INFO="${SIG_INFO:-}"
if [[ -z "$SIG_INFO" ]]; then
  for f in "$RAW"/GSE*_sig_info*.txt* "$RAW"/*sig_info*.txt*; do
    [[ -f "$f" ]] && SIG_INFO="$f" && break
  done
fi
REPURPOSING="${REPURPOSING:-}"
if [[ -z "$REPURPOSING" ]]; then
  for f in "$RAW"/repurposing*.*; do
    [[ -f "$f" ]] && REPURPOSING="$f" && break
  done
fi

# Validate required inputs
if [[ -z "${GCTX:-}" || ! -f "$GCTX" ]]; then
  echo "[err] GCTX not found. Set GCTX or place a *Level5*MODZ*.gctx(.gz) under $RAW" >&2
  exit 1
fi
if [[ -z "${GENE_INFO:-}" || ! -f "$GENE_INFO" ]]; then
  echo "[err] gene_info not found. Set GENE_INFO or place GSE*_gene_info*.txt.gz under $RAW" >&2
  exit 1
fi
if [[ -z "${SIG_INFO:-}" || ! -f "$SIG_INFO" ]]; then
  echo "[err] sig_info not found. Set SIG_INFO or place GSE*_sig_info*.txt.gz under $RAW" >&2
  exit 1
fi

# Show configuration
cat <<CFG
[config]
GCTX        = $GCTX
GENE_INFO   = $GENE_INFO
SIG_INFO    = $SIG_INFO
REPURPOSING = ${REPURPOSING:-<none>}
CHUNK_COLS  = $CHUNK_COLS
PARTITION   = $PARTITION_COL
OUTPUT_DIR  = $OUT_DIR
CFG

# Run conversion
set -x
scperturb-cmap prepare-lincs \
  --gctx "$GCTX" \
  --gene-info "$GENE_INFO" \
  --sig-info "$SIG_INFO" \
  ${REPURPOSING:+--repurposing "$REPURPOSING"} \
  --landmarks \
  --chunk-cols "$CHUNK_COLS" \
  --partition-by "$PARTITION_COL" \
  --output "$OUT_DIR"
set +x

# Validate output
python "$SUBMIT_DIR/scripts/validate_parquet_dataset.py" \
  --dataset "$OUT_DIR" \
  --partition "$PARTITION_COL" \
  --full \
  --head 10 || true

echo "[ok] Conversion complete: $OUT_DIR"
