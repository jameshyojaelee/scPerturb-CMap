#!/usr/bin/env bash
set -euo pipefail

LNX="${1:-/gpfs/commons/home/jameslee/scPerturb-CMap/data/lincs/lincs_level5_landmark_long.parquet}"
PORT="${2:-8501}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate scpc || true
fi

echo "[info] start UI on port $PORT"
echo "[tip] on your laptop run: ssh -L $PORT:localhost:$PORT pe2-login01"

streamlit run src/scperturb_cmap/ui/app.py \
  --server.port "$PORT" --server.headless true \
  -- --demo 0 --lincs "$LNX"

