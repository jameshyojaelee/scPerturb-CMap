# CLI + Streamlit Walkthrough

This tutorial mirrors a lightweight notebook that exercises the CLI (`scperturb_cmap.cli`) and Streamlit UI (`src/scperturb_cmap/ui/app.py`) end-to-end. Follow along in a clean shell, or copy the command cells into your own notebook to keep notes inline.

## 1. Environment and demo assets

```bash
make setup           # creates .venv and installs the project in editable mode
make demo            # writes examples/data/lincs_demo.parquet and a toy target
source .venv/bin/activate
```

The `make demo` target calls into `scripts/demo/print_demo_stats.py`, which seeds a small LINCS library suitable for acceptance tests. All outputs land in `examples/out/`.

## 2. Build a target signature

Use the Typer-based CLI to turn an AnnData file or JSON gene list into a `TargetSignature`. Here we reuse the demo JSON created above:

```bash
scperturb-cmap make-target \
  --h5ad examples/data/demo.h5ad \
  --condition-key condition \
  --case treated \
  --control control \
  --qc-report examples/out/target_qc.html \
  --output examples/out/target.json
```

Key parameters live in `scperturb_cmap.cli:app`; run `scperturb-cmap --help` for the full tree of commands and options.

## 3. Score against the LINCS library

With a target in hand, score it against the demo library using the baseline cosine + GSEA ensemble:

```bash
scperturb-cmap score \
  --target-json examples/out/target.json \
  --library examples/data/lincs_demo.parquet \
  --collapse-replicates \
  --method baseline \
  --top-k 25 \
  --output examples/out/results.parquet
```

The results Parquet includes compound, cell line, MOA, and QC metadata. Swap `--method metric --model-path workspace/artifacts/best.pt` to exercise the DualEncoder flow.

## 4. Explore interactively in Streamlit

Launch the Streamlit app that wraps `scperturb_cmap.api.score.rank_drugs` and the same filtering options as the CLI:

```bash
python -m streamlit run src/scperturb_cmap/ui/app.py -- \
  --lincs examples/data/lincs_demo.parquet \
  --target examples/out/target.json \
  --debug
```

Within the UI you can:
- Upload additional target signatures or H5ADs,
- Inspect QC charts and gene overlap diagnostics,
- Trigger scoring (baseline or metric) with cell line / MOA filters,
- Export enriched result tables back to `examples/out/` for downstream analysis.

## 5. Rerun as a notebook

All commands above can be pasted into a notebook cell (prefix shell cells with `!`) to keep logs next to your exploratory charts. The underlying functions live in `scperturb_cmap.cli` and `scperturb_cmap/ui/app.py`, so you can also import the Python APIs directly:

```python
from pathlib import Path

from scperturb_cmap.api.score import rank_drugs
from scperturb_cmap.io.schemas import TargetSignature
import pandas as pd

target = TargetSignature.model_validate_json(Path("examples/out/target.json").read_text())
library = pd.read_parquet("examples/data/lincs_demo.parquet")
result = rank_drugs(target, library, method="baseline", top_k=10)
print(result.ranking[:3])
```

This notebook-style workflow scales to custom LINCS libraries, larger targets, and metric-model checkpoints—swap paths, set filters, and rerun the cells.
