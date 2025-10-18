# Security & Compliance Review

Date: 2025-10-18  
Reviewed by: Release Automation (ChatGPT acting as security reviewer)

## Dependency Audit

- Command: `module unload OpenSSL/3 || true; module load openssl/1.1; . .venv/bin/activate && pip-audit`
- Result: 8 known vulnerabilities in 6 packages detected by pip-audit:
  - `fastapi 0.104.1` – PYSEC-2024-38 (`>=0.109.1` recommended)
  - `starlette 0.27.0` – GHSA-f96h-pmfr-66vw (`>=0.40.0`), GHSA-2c2j-9gv5-cj73 (`>=0.47.2`)
  - `h11 0.14.0` – GHSA-vqfr-h8mv-ghfj (`>=0.16.0`)
  - `pyarrow 16.1.0` – PYSEC-2024-161 (`>=17.0.0`)
  - `pip 25.2` – GHSA-4xh5-x5gv-qwph (upgrade when patched)
  - `setuptools 58.1.0` (via `pyarrow` build chain) – PYSEC-2022-43012 (`>=65.5.1`), PYSEC-2025-49 (`>=78.1.1`)
- Remediation plan:
  - Align codebase/packaging with current FastAPI/Starlette/H11 releases (target ≥0.110/0.47/0.16 respectively). Requires validating compatibility with our FastAPI app and tests.
  - Upgrade `pyarrow` to ≥17.0.0 (verify acceptance tests and packaging). This will also satisfy the `setuptools` advisory once the new pyarrow pulls a newer build requirement.
  - Track upstream fixes for `pip 25.2`; update venv bootstrap tools once a patched version is available.
  - File follow-up issues for each upgrade to ensure CI coverage before release.

## Licensing Notes

- README and `docs/guides/DATA_NOTES.md` both include the LINCS CC BY 4.0 attribution guidance (Broad Institute, NIH LINCS program, Subramanian et al., Cell 2017) and instructions to avoid re-hosting raw Level 5 data.

## Secrets & Artifacts Hygiene

- `.gitignore` excludes `workspace/`, `examples/out/`, `data/`, and any `*.hydra/` directories (both root-level and recursive), covering the primary generated artifacts and Hydra configs.
- No secrets or credentials are checked into the repository; environment variables are expected to provide database/Redis URLs when needed.

## Follow-up Actions

1. Create work items to upgrade FastAPI/Starlette/H11/pyarrow and retest the API/acceptance suites.
2. Monitor pip release notes for the GHSA-4xh5-x5gv-qwph advisory and update our dev tooling once resolved.
