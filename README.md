# scPerturb-CMap

Single-cell connectivity mapping with deep metric learning for precision drug repurposing.

## Why this matters

- **Bulk vs single-cell**: Traditional connectivity mapping compares drugs to disease signatures from bulk tissue (an average of many mixed cells). That average can miss the rare, aggressive cells that actually drive disease.
- **Precision at the right cells**: This project targets the gene expression of specific cell populations (e.g., therapy-resistant cancer cells or inflammatory immune subsets) to find drugs that act where it matters.
- **Beyond simple matching**: Instead of rank-based opposites, a deep metric learning model learns a shared embedding for cell states and drug perturbations, capturing complex, non-linear reversal patterns.

## What this project does

scPerturb-CMap provides a reference workflow and scaffolding to link single-cell disease signatures to drug perturbation signatures using deep metric learning. The output is a ranked list of candidate drugs most likely to shift a target cell population from a disease state toward a healthy state.

## Workflow

1. **Define the target population**
   - Select the specific cell type or subpopulation from scRNA-seq data; derive its disease signature.

2. **Assemble perturbational references**
   - Load drug-induced expression signatures (e.g., LINCS), harmonize genes, and track metadata (dose, time, cell line).

3. **Train a cross-modal embedding**
   - Use deep metric learning to place cell-state signatures and drug perturbations in a shared space so that reversing drugs are close to disease states they correct.

4. **Score and rank drugs**
   - For the chosen cell population, compute reversal scores in the shared space and produce a top-k list per cell type or cluster.

5. **Validate and iterate**
   - Evaluate with held-out perturbations or benchmarks; refine preprocessing, pair construction, and model hyperparameters.

### Inputs
- Single-cell gene expression with cell-type/cluster labels and case/control grouping.
- Drug perturbation signatures with metadata (compound, dose, time, cell line).

### Outputs
- Ranked candidate drugs per target cell population with reversal scores.
- A shared embedding mapping disease states and perturbations into a common space.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

Early components live under `src/scperturb_cmap/` (e.g., `io/` for schemas/serde and `utils/` for environment helpers). APIs/CLI are evolving.

## Status

Early-stage, under active development. Interfaces and defaults may change as the workflow is finalized.

## License

MIT License. See `LICENSE` for details.
