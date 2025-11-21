# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-09-27
### Added
- Packaged Hydra configs under `scperturb_cmap.configs` so training works from installed wheels.
- New end-to-end workflow and repository layout documentation for release readiness.
- CITATION metadata for academic users (see `CITATION.cff`).
### Ready for 0.2.0 GA
scPerturb-CMap 0.2.0 lands with env-driven API configuration (cache TTLs, request guardrails, LINCS/Redis/Postgres readiness probes), refreshed docs anchored around acceptance runtimes (~30 s with `module load openssl/1.1`), and deployment guidance spanning Docker, Helm, and cloud checklists. Packaging and QA were validated via `make acceptance`, `pytest -q`, a dry-run wheel install, and headless Streamlit smoke tests; outstanding actions include bumping FastAPI/Starlette/H11/pyarrow to close pip-audit advisories, running full Streamlit flows in a browser, and rendering Helm charts in a Helm-enabled environment. Security notes live in `SECURITY.md`, and LINCS CC BY 4.0 attribution remains front and centre so downstream teams stay compliant.

### Changed
- Consolidated generated artifacts under `workspace/` and updated tooling to use environment Python.
- Bumped project version metadata to 0.2.0.
- Documented that `make acceptance` remains a local-only smoke test.
- Explainability contributions now mirror the active scoring method (cosine+GSEA or metric) and rescale to the reported score so narratives and waterfalls match the ranked outputs.

### Fixed
- Acceptance harness now respects the active Python interpreter instead of assuming `.venv` paths.

### CI/CD and Releases
- GitHub Actions pipeline (`ci.yml`) now runs `make lint`, `make test`, and `make acceptance` across Python 3.10–3.12, reusing a cached `.venv` with the optional Redis/PyArrow extras enabled so API checks remain non-blocking.
- Documentation is validated via `mkdocs build --strict`, with the rendered site and built distributions uploaded as workflow artifacts.
- Tagged releases (`v*`) reuse the same build job to produce wheels/sdists from `pyproject.toml` metadata and publish to PyPI using `pypa/gh-action-pypi-publish`; non-tag pushes and PRs skip publishing but still surface artifacts for review.
- The project README now advertises the CI status badge so contributors can quickly verify pipeline health.

## [0.1.1] - 2025-09-24
### Added
- Initial public release of scPerturb-CMap.
- CLI for LINCS ingestion, target construction, scoring, training, and UI.
- Baseline (cosine + GSEA) connectivity with z- and p-values.
- Streamlit UI with QC, MOA enrichment, and exports.
- Acceptance harness and tests.

### Fixed
- pyproject.toml PEP 621 fields (remove invalid inline urls table and extraneous field) for CI build.
