# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-09-27
### Added
- Packaged Hydra configs under `scperturb_cmap.configs` so training works from installed wheels.
- New end-to-end workflow and repository layout documentation for release readiness.
- CITATION metadata for academic users (see `CITATION.cff`).
### Ready for 0.2.0 GA
- Documented LINCS licensing/attribution requirements and linked the CLUE distribution portal.
- Recorded `make acceptance` runtime expectations (with OpenSSL module guidance) and validated the target on current hardware.
- Launched the Streamlit UI headlessly to confirm exports/bookmarks still initialise without errors—no visual changes required, so screenshots remain intact.
- Ran `codespell` across README/docs (ignoring domain-specific terms) and cleaned the only hit.
- Pruned stray Hydra outputs under `examples/out/previous_runs/` and tightened `.gitignore` for transient run folders.

### Changed
- Consolidated generated artifacts under `workspace/` and updated tooling to use environment Python.
- Bumped project version metadata to 0.2.0.
- Documented that `make acceptance` remains a local-only smoke test.

### Fixed
- Acceptance harness now respects the active Python interpreter instead of assuming `.venv` paths.

## [0.1.1] - 2025-09-24
### Added
- Initial public release of scPerturb-CMap.
- CLI for LINCS ingestion, target construction, scoring, training, and UI.
- Baseline (cosine + GSEA) connectivity with z- and p-values.
- Streamlit UI with QC, MOA enrichment, and exports.
- Acceptance harness and tests.

### Fixed
- pyproject.toml PEP 621 fields (remove invalid inline urls table and extraneous field) for CI build.
