# Case Study Workflows

The `examples/case_studies/` directory collects the reproducible tutorials that
shipped with earlier releases of scPerturb-CMap. Each subdirectory contains:

- `CASE_STUDY_*.md` – narrative walkthrough of the analysis
- `scripts/run_analysis.py` – Typer CLI that rebuilds the results
- `results/` – sample outputs produced by the CLI

| Case | Path | Focus |
| --- | --- | --- |
| NSCLC CD8<sup>+</sup> exhaustion | `nsclc_cd8/` | Reversing T cell exhaustion in lung cancer |
| EMT in TNBC | `emt_breast/` | Repressing epithelial–mesenchymal transition signatures |
| IFN-high macrophages | `ifn_macrophages/` | Dampening interferon-driven inflammation |

## Running a Case Study

Execute the bundled CLI to regenerate rankings against the demo LINCS library:

```bash
python examples/case_studies/nsclc_cd8/scripts/run_analysis.py \
  --library examples/data/lincs_demo.parquet \
  --results-dir /tmp/nsclc_results
```

Swap `nsclc_cd8` for `emt_breast` or `ifn_macrophages` to run the other demos.
The scripts accept any `TargetSignature` JSON you produce with
`scperturb-cmap make-target`, so you can also treat them as templates for your
own analyses.

## Copying the Template

To start a new study from one of these folders:

```bash
cp -r examples/case_studies/nsclc_cd8 examples/case_studies/my_project
```

Then edit `scripts/run_analysis.py` to point at your target signature and LINCS
library paths.

## Related Docs

- [Quickstart](../../docs/quickstart.md)
- [Case study overview](../../docs/cases/index.md)
- [Explainability guide](../../docs/explainability.md)
