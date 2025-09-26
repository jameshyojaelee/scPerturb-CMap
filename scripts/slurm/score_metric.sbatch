#!/bin/bash
#SBATCH -J score_metric
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o %x.%j.out
set -euo pipefail

# Prefer Conda env 'scpc' if available; otherwise try local venv
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate scpc || true
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

PROJ="/gpfs/commons/home/jameslee/scPerturb-CMap"
LINCS="$PROJ/data/lincs/lincs_level5_landmark_long.parquet"
TARGET_JSON="$PROJ/examples/out/target_sig_real.json"
FILTERED_TARGET_JSON="$PROJ/examples/out/target_sig_filtered_metric.json"
MODEL="$PROJ/workspace/artifacts/best.pt"
OUT="$PROJ/examples/out/results_metric.parquet"

echo "[score-metric] target_json=$TARGET_JSON lincs=$LINCS model=$MODEL"

if [[ ! -f "$TARGET_JSON" ]]; then
  echo "[err] Target JSON not found: $TARGET_JSON" >&2
  exit 1
fi
if [[ ! -f "$LINCS" ]]; then
  echo "[err] LINCS file not found: $LINCS" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "[err] Checkpoint not found: $MODEL" >&2
  exit 1
fi

# Filter target genes to library genes
python - <<PY
import json, pandas as pd
lib_path = "$LINCS"
target_in = "$TARGET_JSON"
target_out = "$FILTERED_TARGET_JSON"
lib = pd.read_parquet(lib_path, engine='pyarrow')
lib_genes = set(lib['gene_symbol'].astype(str).str.upper().unique())
ts = json.load(open(target_in))
fg, fw = [], []
for g, w in zip(ts['genes'], ts['weights']):
    if str(g).upper() in lib_genes:
        fg.append(str(g))
        fw.append(float(w))
if not fg:
    raise SystemExit('[err] No overlap between target and library genes')
json.dump({'genes': fg, 'weights': fw, 'metadata': ts.get('metadata', {})}, open(target_out, 'w'))
print(f"[ok] filtered target -> {target_out} (n={len(fg)})")
PY

scperturb-cmap score \
  --target-json "$FILTERED_TARGET_JSON" \
  --library     "$LINCS" \
  --method metric \
  --model-path "$MODEL" \
  --top-k 200 \
  --output "$OUT"

python - <<'PY'
import pandas as pd, os
p=os.environ.get('OUT_PATH','') or "/gpfs/commons/home/jameslee/scPerturb-CMap/examples/out/results_metric.parquet"
if os.path.exists(p):
    df=pd.read_parquet(p); print(df.head()); print('[ok] rows=', len(df))
else:
    print('[err] results file missing:', p)
PY
