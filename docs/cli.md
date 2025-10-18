# Command-Line Interface

Run `scperturb-cmap --help` to see the top-level commands. The most common
subcommands are summarised below with example invocations.

## `prepare-lincs`

Convert raw LINCS Level 5 data (GCTX or long-form Parquet/CSV/TSV) into the
long-format table used throughout the toolkit.

```bash
scperturb-cmap prepare-lincs \
  --gctx data/raw/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ.gctx \
  --gene-info data/raw/GSE92742_Broad_LINCS_gene_info.txt \
  --sig-info data/raw/GSE92742_Broad_LINCS_sig_info.txt.gz \
  --repurposing data/raw/repurposing_drugs_20180907.txt \
  --landmarks \
  --partition-by cell_line \
  --output data/lincs/lincs_level5_landmark_long
```

- `--landmarks` restricts to the 978 landmark genes.
- Use `--input` instead of `--gctx` for already flattened tables.
- Partitioning by `cell_line` keeps downstream reads fast.

## `make-target`

Derive a `TargetSignature` JSON file from an `.h5ad` dataset or curated gene
lists.

```bash
scperturb-cmap make-target \
  --h5ad data/sc/nsclc_cd8.h5ad \
  --cluster-key leiden \
  --cluster exhausted_cd8 \
  --reference rest \
  --output examples/out/nsclc_target.json \
  --qc-report examples/out/nsclc_target_qc.json
```

Key flags:

- `--pseudobulk-key` groups cells before computing contrasts.
- `--library-genes` (text file) reports overlap with a specific LINCS library.
- `--up-genes` / `--down-genes` accept curated gene lists when no `.h5ad`
  exists.

## `score`

Score a target signature against a LINCS library.

```bash
scperturb-cmap score \
  --target-json examples/out/nsclc_target.json \
  --library data/lincs/lincs_level5_landmark_long \
  --method baseline \
  --top-k 50 \
  --collapse-replicates \
  --output examples/out/results_baseline.parquet
```

- Set `--method metric --model-path workspace/artifacts/best.pt` to blend the
  DualEncoder.
- `--cell-line`, `--moa`, `--dose-range` filter the library on the fly.
- `--json-output` emits results + metadata as JSON.

## `power`

Subcommands for signature stability and rank confidence analysis.

```bash
scperturb-cmap power sample-size \
  --h5ad data/sc/nsclc_cd8.h5ad \
  --cluster-key leiden \
  --cluster exhausted_cd8 \
  --reference rest \
  --summary-output workspace/reports/sample_size.csv
```

Other modes:

- `power min-cells` – recommended cells per cluster
- `power rank-ci` – bootstrap confidence intervals for rankings
- `power stability` – replicate signature stability metrics
- `power permutation-test` – permutation p-values for score differences

## `train`

Kick off metric-model training using Hydra configs.

```bash
scperturb-cmap train \
  pairs_path=examples/out/metric_dataset/metric_pairs.parquet \
  targets_path=examples/out/metric_dataset/metric_targets.jsonl \
  library_path=examples/out/metric_dataset/metric_library.parquet \
  epochs=10 \
  hydra.run.dir=workspace/runs/demo-metric
```

- Provide real inversion pairs to fine-tune the DualEncoder on custom data.
- Outputs checkpoints under `workspace/artifacts/`.

## `evaluate`

Compute offline metrics for a saved metric-model checkpoint.

```bash
scperturb-cmap evaluate --checkpoint workspace/artifacts/best.pt
```

## `ui`

Launch the Streamlit UI.

```bash
scperturb-cmap ui
```

- By default the app reads `examples/data/lincs_demo.parquet`.
- Set `SCPC_LINCS` in your environment to point at a larger library.

## `validate-h5ad`

Perform quick sanity checks on `.h5ad` files before target creation.

```bash
scperturb-cmap validate-h5ad \
  --h5ad data/sc/nsclc_cd8.h5ad \
  --expect-genes L1000
```

Reports symbol coverage, overlap with a reference gene list, and highlights
candidate `obs` columns for clustering or pseudobulk aggregation.
