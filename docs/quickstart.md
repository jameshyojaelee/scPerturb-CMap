# Quickstart

Follow these steps to install scPerturb-CMap, produce your first ranked list, and
explore the Streamlit UI export/bookmark features.

## Installation

```bash
git clone https://github.com/jameslee/scPerturb-CMap.git
cd scPerturb-CMap
make setup
```

The `make setup` target creates a `.venv/` directory and installs the project in
editable mode with development dependencies.

Prefer Conda? Use the provided environment spec instead:

```bash
conda env create -f environment.yml
conda activate scperturb-cmap
```

## First Analysis (CLI)

Generate the demo assets and score the synthetic target against the bundled
LINCS subset.

```bash
make demo

scperturb-cmap score \
  --target-json examples/out/target.json \
  --library examples/data/lincs_demo.parquet \
  --method baseline \
  --top-k 50 \
  --output examples/out/results.parquet
```

The resulting Parquet file contains the top hits with z-scores, p-values, and
mechanism-of-action annotations. Acceptance tests (`python3 -m
scripts.check_acceptance`) run the same workflow and verify that the metric
model improves recall.

## Streamlit UI Exports & Bookmarks

Launch the UI with `make ui` (or `scperturb-cmap ui`) and open
`http://localhost:8501/`.

- **Exports** – after scoring, use the *Download → CSV* or *Download → Parquet*
  buttons in the results tab. Files are written to
  `workspace/ui_exports/` with timestamped folders.
- **Bookmarking** – configure the target, filters, and blend settings, then copy
  the URL displayed in the *Bookmark* panel. Visiting the saved link restores
  the UI state so collaborators see the same configuration.
- **Preset management** – saved presets live in
  `examples/data/ui_presets.json`. Edit the file or import/export presets from
  the sidebar.

The UI reads the LINCS path from `SCPC_LINCS`; set this environment variable
before launching to point at a larger library.

## Acceptance Checks

Run `make acceptance` after setup to exercise the bundled demo workflow (baseline scoring plus the
metric fine-tune). On the reference workstation it completes in ~30 seconds and writes temporary
artifacts under `workspace/`. On clusters with environment modules, load `openssl/1.1` before
invoking the target (e.g., `module unload OpenSSL/3 && module load openssl/1.1 && make acceptance`)
so PyArrow and Streamlit have access to the expected OpenSSL runtime.

## Next Steps

- Work through the case-study scripts under `examples/case_studies/*/scripts/run_analysis.py`.
- Fine-tune the DualEncoder with `scperturb-cmap train`.
- Read the [CLI reference](cli.md) and [API reference](api.md) for additional
  automation hooks.
