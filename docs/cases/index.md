# Case Study Overview

The case studies show end-to-end analyses using real disease signatures. Each
folder pairs input data, scripts, and results so you can reproduce the published
rankings. Use the `run_analysis.py` helper in each `scripts/` directory to
execute the workflow.

## Available Case Studies

| Case study | Highlights | Entry point |
| --- | --- | --- |
| NSCLC CD8+ T cell exhaustion | Reversal of exhausted tumour-infiltrating lymphocytes | `case_studies/nsclc_cd8/scripts/run_analysis.py` |
| EMT in triple-negative breast cancer | EMT reversal with pathway enrichment and validation plan | `case_studies/emt_breast/scripts/run_analysis.py` |
| IFN-high macrophages | Cell-line-aware predictions with uncertainty estimates | `case_studies/ifn_macrophages/scripts/run_analysis.py` |

## Running a Case Study

```bash
python case_studies/nsclc_cd8/scripts/run_analysis.py \
  --target-json case_studies/nsclc_cd8/data/target.json \
  --library data/lincs/lincs_level5_landmark_long \
  --metric-model workspace/artifacts/best.pt \
  --top-k 100
```

Tips:

- Populate the `data/` directory with your processed `.h5ad` outputs and
  exported targets.
- The scripts write results and QC summaries to the `results/` directory.
- Pass `--metric-model` only when a trained DualEncoder checkpoint is available.

Refer to the individual Markdown files in `docs/cases/` for biological context
and interpretation notes.

