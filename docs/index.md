# scPerturb-CMap

Single-cell connectivity mapping toolkit for drug repurposing. The platform blends a fast cosine+GSEA baseline with a DualEncoder metric model trained on LINCS inversions, ships a Streamlit UI, and provides CLI/Python APIs for end-to-end analysis.

## Quick Links

- **Install**: `pip install scperturb-cmap`
- **CLI help**: `scperturb-cmap --help`
- **Streamlit UI**: `make ui`
- **Quickstart guide**: [docs/quickstart](quickstart.md#acceptance-checks)
- **API reference**: [docs/api](api.md)

## Release Readiness At a Glance

- `make acceptance` (with `module load openssl/1.1`) executes the bundled demo workflow in ~30 seconds and reports a metric recall lift of ~0.83 over the baseline.
- `pytest -q` (run with `PYTHONPATH=.` inside the project virtualenv) completes in ~17 seconds on the reference workstation.
- Streamlit exports/bookmarks are verified headlessly; the UI reflects the same visuals documented in the guides.

Refer to the [Quickstart Acceptance Checks](quickstart.md#acceptance-checks) for the exact shell invocations.

## LINCS Data Licensing & Attribution

The demo builds bundle small LINCS L1000 excerpts purely for smoke tests. Production work must source the official archives from the [Broad Institute Connectivity Map (CLUE) portal](https://clue.io/connectopedia/data_download). The dataset is released under Creative Commons CC BY 4.0; when redistributing derived subsets you must:

1. Attribute the Broad Institute, NIH LINCS program, and the original publications (e.g., Subramanian *et al.*, Cell 2017).
2. Preserve the original license text and direct collaborators to obtain authoritative copies from CLUE whenever possible (avoid re-hosting raw Level 5 files in public buckets).

## Where to Next?

- Build your first analysis via the [Quickstart](quickstart.md).
- Dive into tooling in the [CLI reference](cli.md) and [API reference](api.md).
- Explore case studies under `case_studies/` or the [Case Studies](cases/index.md) section.
- Review deployment options in the [Cloud guide](deployment/CLOUD_DEPLOYMENT.md) and accompanying Docker/Helm docs.
