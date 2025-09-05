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
FILTERED_TARGET_JSON="$PROJ/examples/out/target_sig_filtered.json"

echo "[score] target_json=$TARGET_JSON lincs=$LINCS"

if [[ ! -f "$TARGET_JSON" ]]; then
  echo "[err] Target JSON not found: $TARGET_JSON" >&2
  exit 1
fi
if [[ ! -f "$LINCS" ]]; then
  echo "[err] LINCS file not found: $LINCS" >&2
  exit 1
fi

# Filter target to only genes present in the LINCS library (avoid low-overlap error)
python - <<PY
import json, pandas as pd, os
lib_path = os.environ.get('LINCS')
target_in = os.environ.get('TARGET_JSON')
target_out = os.environ.get('FILTERED_TARGET_JSON')

lib = pd.read_parquet(lib_path, engine='pyarrow')
lib_genes = set(lib['gene_symbol'].astype(str).str.upper().unique())

ts = json.load(open(target_in))
genes = [str(g) for g in ts['genes']]
weights = ts['weights']
pairs = [(g, w) for g, w in zip(genes, weights) if g.upper() in lib_genes]
if not pairs:
    raise SystemExit('[err] No overlap between target and library genes')
fg, fw = zip(*pairs)
filtered = { 'genes': list(fg), 'weights': list(fw), 'metadata': ts.get('metadata', {}) }
with open(target_out, 'w') as f:
    json.dump(filtered, f)
print(f"[ok] filtered target -> {target_out} (n={len(fg)})")
PY

scperturb-cmap score \
  --target-json "$FILTERED_TARGET_JSON" \
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

