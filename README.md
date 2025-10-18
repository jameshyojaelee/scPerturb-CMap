# scPerturb-CMap

The Broad Connectivity Map (LINCS L1000) captures millions of empirically measured drug responses. scPerturb-CMap is the bridge that lets single-cell researchers mine that atlas without rebuilding perturbation screens from scratch. Starting with a troublesome or rare cell population (e.g., EMT-like tumor cells, IFN-high macrophages, exhausted T cells), you derive a gene signature and immediately query the L1000 treasury for compounds proven to push cells in the opposite direction. In place of months-long screening campaigns, you generate ordered, testable drug hypotheses the same day—complete with effect sizes, z-scores, p-values, QC stats, and mechanism-of-action enrichments.

scPerturb-CMap blends a fast statistical baseline with a learned DualEncoder metric model trained on known inversion pairs. This machine learning component learns embeddings for targets and perturbations and is blended with the baseline at inference to sharpen ranking accuracy.

Why use scPerturb-CMap:
1. **Targets the “undruggable.”** Rare states no longer disappear in bulk averages; ranking is driven by the exact cluster causing pathology.
2. **Leverages a decade of data.** Mine the public L1000 archive before expensive single-cell perturbation experiments.
3. **Accelerates repurposing.** Connect patient-derived or experimental signatures to approved/investigational compounds with immediate readouts for bench validation.
4. **Democratises analysis.** Fits existing workflows (AnnData `.h5ad`, curated gene lists, LINCS Parquet), and ships with CLI, Python API, and Streamlit UI for mixed computational/experimental teams.

#### How scPerturb-CMap differs from scGen / scPerturb

| Feature | **scGen / scPerturb** | **scPerturb-CMap** |
| :-- | :-- | :-- |
| Primary question | *“If I apply perturbation X, what will my single-cell transcriptomes look like?”* | *“Given this disease signature, which known compounds have been observed to reverse it?”* |
| Core capability | Learns perturbation rules within a reference single-cell dataset and predicts new responses. | Searches the external LINCS L1000 atlas to retrieve real perturbations ranked by inversion strength. |
| Required inputs | A single-cell experiment that already contains the perturbation of interest (treated vs control). | A target signature from scRNA-seq (`.h5ad`) or curated up/down gene lists; optional custom LINCS-style libraries. |
| Outputs | Simulated single-cell expression profiles under hypothetical perturbations. | Ranked list of real compounds with connectivity scores, z/p statistics, QC metrics, and MOA enrichment. |
| Analogy | **Flight simulator** – models how a plane behaves under new conditions. | **Flight search engine** – scans all existing routes to find the optimal therapeutic “flight” toward reversal. |

The package ships with:

- a fast baseline (cosine + GSEA ensemble) that emits z-scores and p-values,
- a DualEncoder metric model that can be trained on real inversion pairs,
- CLI utilities for LINCS ingestion, target construction, scoring, and training,
- a Streamlit UI for interactive analysis.


## Concept Overview

Traditional connectivity mapping averages bulk transcriptomes and may miss rare cell states. scPerturb-CMap instead works with single-cell targets.

![Fig 1. scPerturb-CMap pipeline](figs/fig1_system_diagram_r.png)

*Figure 1. Single-cell targets align to curated L1000 signatures before the baseline and DualEncoder branches blend into a ranked compound readout.*

## Feature Highlights

- **Baseline ensemble** – cosine connectivity + GSEA, exported with z-scores and double-sided p-values.
- **Metric learning** – DualEncoder trained with NT-Xent or triplet loss on real or synthetic inversion pairs; blended with the baseline at inference.
- **Replicate-aware preprocessing** – optional MODZ collapsing (`--collapse-replicates`) when `replicate_id` is present.
- **Target engineering** – pseudobulk grouping (`--pseudobulk-key`), QC summaries (gene balance, overlap with LINCS), and JSON/CSV exports.
- **Pair generation helpers** – utilities under `scperturb_cmap.data.pairs` to sample positives/negatives from LINCS metadata.
- **Rich analytics** – Streamlit UI exposing target QC, MOA enrichment bars, and cell-line heatmaps.

---

## Installation

```bash
pip install scperturb-cmap
```

To hack on the project locally:

```bash
git clone https://github.com/jameslee/scPerturb-CMap.git
cd scPerturb-CMap
make setup
# OR create the Conda environment with dev extras
conda env create -f environment.yml
conda activate scperturb-cmap
```

Generated artifacts default to the `workspace/` directory (checkpoints under `workspace/artifacts/`, logs under `workspace/logs/`, documentation builds under `workspace/site/`). Point environment variables (e.g., `SCPC_MODEL`) there when running from a fresh install.

---

## LINCS Data Licensing & Attribution

The demo bundles include small LINCS L1000 excerpts to keep acceptance tests fast. Production use
still relies on the official LINCS distribution hosted by the [Broad Institute Connectivity Map
(CLUE) team](https://clue.io/connectopedia/data_download). The data are released under a Creative
Commons CC BY 4.0 license; downstream analysis must (a) attribute the Broad Institute, the NIH
LINCS program, and the original publications (e.g. Subramanian *et&nbsp;al.*, Cell 2017) and
(b) preserve the original license statement when you redistribute derived subsets. scPerturb-CMap
does not grant additional rights to the LINCS measurements—ensure collaborators download
authenticated archives directly from CLUE when possible and avoid re-hosting raw Level 5 files in
public buckets.

---

## Quickstart

```bash
# Create a local virtual environment and install the package + dev extras
make setup

# Generate synthetic demo LINCS + AnnData assets
make demo

# Score the demo target against the demo library (writes examples/out/results.parquet)
scperturb-cmap score \
  --target-json examples/out/target.json \
  --library examples/data/lincs_demo.parquet \
  --collapse-replicates \
  --method baseline \
  --top-k 50 \
  --output examples/out/results.parquet

# Launch the Streamlit dashboard
make ui

# Run short synthetic training + evaluation loops
make train
make evaluate

# Developer hygiene
make lint
make test
```

> **Python**: 3.10+ is required. All commands above assume GNU Make and a POSIX shell.


1. **Prepare the library** – Convert or download LINCS L1000 signatures into a long-form Parquet/CSV table (`scperturb-cmap prepare-lincs`) and keep them under `data/lincs/` or another Arrow-friendly location.
2. **Build a target signature** – Derive up/down weights from an AnnData object or curated gene lists using `scperturb-cmap make-target` (or the Streamlit UI). Inspect QC summaries before moving on.
3. **Score compounds** – Run `scperturb-cmap score` with `--method baseline` for a fast cosine+GSEA ensemble or pass `--method metric --model-path workspace/artifacts/best.pt` to blend in the trained DualEncoder.
4. **(Optional) Train the metric model** – Supply curated inversion pairs via `scperturb-cmap train` (Hydra config under `configs/train.yaml`) to refine the DualEncoder checkpoint written to `workspace/artifacts/`.
5. **Explore interactively** – Launch `make ui` to open the Streamlit dashboard, reuse existing targets, and export ranked hypotheses with MOA enrichment plots for bench scientists.
6. **Validate & automate** – Use `make acceptance` for smoke tests, `make lint`/`make test` in CI, and the HPC scripts under `scripts/slurm/` for batch jobs.

> `make acceptance` is designed as a local smoke test and is not executed as part of the default CI workflows.


**Recall@50 by cell line**

![Fig 2. Recall@50 by cell line](figs/fig2_recall_by_cell_line_r.png)

*Figure 2. Baseline connectivity (blue) versus the blended metric model (orange) across reference cell lines with 95% confidence intervals.*

**MoA enrichment landscape**

![Fig 3. MoA enrichment bar chart](figs/fig3_moa_enrichment_r.png)

*Figure 3. Mechanism-of-action enrichment highlighting up- and down-regulated classes by log2 change and significance.*

**Case study ranking trace**

![Fig 4. Case study ranked compounds](figs/fig4_case_study_rank_plot_r.png)

*Figure 4. Blended connectivity scores for the top-ranked compounds in the NSCLC case study, with annotations for the leading hits.*

See **[STRUCTURE.md](STRUCTURE.md)** for complete directory tree and navigation guide.

| Command | Purpose |
| --- | --- |
| `scperturb-cmap make-target` | Build a target signature from `.h5ad` clusters or explicit gene lists. Options include `--pseudobulk-key`, `--qc-report`, and `--library-genes` to capture QC context. |
| `scperturb-cmap prepare-lincs` | Convert Level 5 GCTX to long-form LINCS tables, apply landmark filters, join MOA/target annotations, and optionally partition by `cell_line`. |
| `scperturb-cmap score` | Score a target against a LINCS library using `baseline` or `metric` methods. Supports rich filtering (`--cell-line(s)`, `--moa(s)`, `--dose-range`, `--touchstone`), replicate collapsing, and Parquet output. |
| `scperturb-cmap train` | Train the DualEncoder. With `pairs_path`, `targets_path`, and `library_path` the trainer uses real inversion data; otherwise, it falls back to synthetic toy data. |
| `scperturb-cmap device` / `scperturb-cmap diagnose` | Quick checks for device availability and environment diagnostics. |

Python APIs mirror the CLI; see `src/scperturb_cmap` for modules such as `api.score`, `data.pairs`, and `models.train`.


1. **Assemble positives**: create a table with at least `target_id` and `signature_id`. Use `prepare_pair_table(...)` to attach negatives or supply a `label` column (1 = inversion, 0 = non-inversion).
2. **Export target JSON Lines**: each record must include `target_id`, `genes`, and `weights`. The CLI generator (`make-target --qc-report`) can write both the JSON target and a QC summary.
3. **Train**:

```bash
scperturb-cmap train \
  pairs_path=/path/to/pairs.parquet \
  targets_path=/path/to/targets.jsonl \
  library_path=/path/to/lincs_long.parquet \
  negatives_per_target=5 \
  epochs=10 \
  batch_size=128
```

The trainer auto-infers the gene dimension, logs metrics in `workspace/artifacts/metrics.json`, and writes `workspace/artifacts/best.pt`. You can point scoring runs to that checkpoint via `--method metric --model-path workspace/artifacts/best.pt`.


Use the built-in converter when you have raw Level 5 assets:

```bash
# Optional landmark extraction
scperturb-cmap landmarks \
  --gene-info /path/to/gene_info.txt \
  --output data/l1000_landmarks.txt

# GCTX ➜ Parquet (partitioned by cell_line for predicate pushdown)
scperturb-cmap prepare-lincs \
  --gctx /path/to/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ.gctx \
  --gene-info /path/to/gene_info.txt \
  --sig-info /path/to/GSE92742_Broad_LINCS_sig_info.txt.gz \
  --repurposing /path/to/repurposing_drugs.tsv \
  --landmarks \
  --partition-by cell_line \
  --output data/lincs/lincs_level5_landmark_long
```

Tips:
- Supply `--landmarks-file` to reuse an existing 978-gene list; otherwise the converter derives one.
- For very large libraries, prefer `--partition-by cell_line` and use `--cell-lines` during scoring to leverage Arrow predicate pushdown.
- A validation script (`python scripts/validators/validate_parquet_dataset.py --dataset …`) summarizes partition counts and schema consistency.

![Streamlit UI](docs/assets/ui.png)
`make ui` launches a browser app that:
- loads the demo LINCS table by default (override with `--lincs <path>` or `SCPC_LINCS`),
- allows target creation from gene lists or uploaded `.h5ad` files,
- visualises the target signature and QC metrics,
- runs scoring (baseline or metric) with the same filtering options as the CLI,
- displays ranked results alongside MOA enrichment bars and heatmaps,
- supports CSV/JSON exports.

Example top-20 predicted inversions (NSCLC CD8+ T target):
![Top-20 NSCLC](docs/assets/top20_nsclc.png)


The project defines three acceptance checks:
1. Baseline scoring on the demo completes in <60 seconds and emits z-scores/p-values.
2. A short DualEncoder training run improves recall@50 by ≥10 percentage points over the untrained model.
3. The Streamlit UI can load the demo dataset and export ranked results.

Run them together:

```bash
make acceptance
```

The script scores the demo, materialises `examples/out/metric_dataset/` (synthetic but structured like real inversion pairs), trains the DualEncoder against those files, and ensures recall@5 improves by ≥10 percentage points.

Cluster-specific setup, Slurm examples, and environment hints live in [`docs/hpc.md`](docs/hpc.md). In short:
- `make hpc-setup` provisions an environment (Conda if available, otherwise venv).
- `scripts/*.sbatch` provide job templates for data conversion, scoring, training, and UI tunnels.
- Respect site-specific module requirements (e.g., load CUDA before launching GPU jobs).

If you use scPerturb-CMap in your research, please cite:

```bibtex
@software{scperturb_cmap2025,
  author = {Lee, James and contributors},
  title = {scPerturb-CMap: Single-Cell Connectivity Mapping for Drug Repurposing},
  year = {2025},
  version = {0.2.0},
  url = {https://github.com/jameslee/scPerturb-CMap},
  doi = {10.5281/zenodo.1234567}
}
```

### Related Work

scPerturb-CMap builds on and complements these foundational tools:

- **LINCS L1000**: Subramanian et al. (2017) *Cell* - The perturbation atlas
- **Connectivity Map**: Lamb et al. (2006) *Science* - Original connectivity mapping concept
- **scGen**: Lotfollahi et al. (2019) *Nature Methods* - Single-cell perturbation prediction
- **Scanpy**: Wolf et al. (2018) *Genome Biology* - Single-cell analysis framework


scPerturb-CMap includes a comprehensive explainability framework providing SHAP-like interpretability:

- **Gene-level attribution**: Which specific genes drive each drug's ranking
- **Waterfall plots**: Visual gene-by-gene contribution breakdown  
- **Pathway enrichment**: GO/KEGG/Reactome integration with network visualization
- **Automated narratives**: Human-readable explanations citing specific gene inversions
- **Cell-line-specific predictions**: With bootstrap confidence intervals
- **Comparison mode**: Explains why Drug A ranks higher than Drug B


Three comprehensive real-world case studies with full workflow examples:

### 1. NSCLC CD8+ T Cell Exhaustion
Reversing immune exhaustion with top-20 validated compounds and literature citations.  
**See**: [case_studies/nsclc_cd8/CASE_STUDY_NSCLC_CD8.md](case_studies/nsclc_cd8/CASE_STUDY_NSCLC_CD8.md)

### 2. EMT in Triple-Negative Breast Cancer
EMT reversal with MOA pathway analysis and experimental validation plan.  
**See**: [case_studies/emt_breast/CASE_STUDY_EMT_BREAST.md](case_studies/emt_breast/CASE_STUDY_EMT_BREAST.md)

### 3. IFN-High Macrophages
Cell-line-specific predictions for inflammatory disease with confidence intervals.  
**See**: [case_studies/ifn_macrophages/CASE_STUDY_IFN_MACROPHAGES.md](case_studies/ifn_macrophages/CASE_STUDY_IFN_MACROPHAGES.md)

Each case study includes:
- Complete .h5ad → ranked drugs workflow
- QC interpretation and validation
- Suggested experimental validations
- Literature citations
- MOA enrichment analysis
