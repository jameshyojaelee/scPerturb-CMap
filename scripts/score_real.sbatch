#!/bin/bash
#SBATCH -J score_real
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o %x.%j.out

set -euo pipefail

# Prefer Conda env 'scpc' if available; otherwise try local venv
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate scpc || true
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

# Allow overriding base via SCPC_BASE; default to example path
PROJ="${SCPC_BASE:-/gpfs/commons/home/jameslee/scPerturb-CMap}"
LINCS="$PROJ/data/lincs/lincs_level5_landmark_long.parquet"
# TargetSignature JSON expected by CLI
TARGET_JSON="$PROJ/examples/out/target_sig_real.json"
OUT="$PROJ/examples/out/results_real.parquet"

echo "[score] target_json=$TARGET_JSON lincs=$LINCS"

if [[ ! -f "$TARGET_JSON" ]]; then
  echo "[err] Target JSON not found: $TARGET_JSON" >&2
  exit 1
fi
if [[ ! -f "$LINCS" ]]; then
  echo "[err] LINCS file not found: $LINCS" >&2
  exit 1
fi

scperturb-cmap score \
  --target-json "$TARGET_JSON" \
  --library     "$LINCS" \
  --method baseline \
  --top-k 200 \
  --output "$OUT"

python - <<'PY'
import pandas as pd, os
p=os.environ.get('OUT_PATH','') or "/gpfs/commons/home/jameslee/scPerturb-CMap/examples/out/results_real.parquet"
if os.path.exists(p):
    df=pd.read_parquet(p); print(df.head()); print('[ok] rows=', len(df))
else:
    print('[err] results file missing:', p)
PY

