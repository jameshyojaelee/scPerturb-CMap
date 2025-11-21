# Case Study Overview

The case studies show end-to-end analyses using real disease signatures. Each
folder pairs input data, scripts, and results so you can reproduce the published
rankings. Use the `run_analysis.py` helper in each `scripts/` directory to
execute the workflow.

## Available Case Studies

| Case study | Highlights | Entry point |
| --- | --- | --- |
| NSCLC CD8+ T cell exhaustion | Reversal of exhausted tumour-infiltrating lymphocytes | `examples/case_studies/nsclc_cd8/scripts/run_analysis.py` |
| EMT in triple-negative breast cancer | EMT reversal with pathway enrichment and validation plan | `examples/case_studies/emt_breast/scripts/run_analysis.py` |
| IFN-high macrophages | Cell-line-aware predictions with uncertainty estimates | `examples/case_studies/ifn_macrophages/scripts/run_analysis.py` |

## Running a Case Study

```bash
python examples/case_studies/nsclc_cd8/scripts/run_analysis.py \
  --target-json path/to/target.json \
  --library examples/data/lincs_demo.parquet \
  --metric-model workspace/artifacts/best.pt \
  --top-k 100
```

Tips:

- Store your processed targets wherever you prefer and pass the path via
  `--target-json`.
- The scripts write results and QC summaries to the `results/` directory.
- Pass `--metric-model` only when a trained DualEncoder checkpoint is available.

Refer to the individual Markdown files in `docs/cases/` for biological context
and interpretation notes.

## Benchmark harness

Use `scperturb-cmap benchmark` (or `python scripts/benchmarks/run_benchmarks.py`) to run a lightweight
comparison between the scPerturb-CMap baseline scorer and a random baseline on the reproducible
dataset under `examples/data/benchmark_synthetic.csv`. Outputs include:

- `results/benchmarks/benchmark_results.json` capturing recall@k and precision@k.
- `results/benchmarks/benchmark_metrics.html` with grouped bars for each metric/method pair.

Interpretation: the synthetic target is constructed so `sig_inverter` should rank first; recall@1
for scPerturb-CMap should exceed the random baseline. Extend the harness with additional methods
(e.g., scGen, scDE) by augmenting `scperturb_cmap/benchmarking.py` and pointing the CLI to a larger
benchmark dataset.
