

# Prompt 3 – Packaging and CI Alignment

Context: `/gpfs/commons/home/jameslee/scPerturb-CMap`

Goals:
- Update `recipe/meta.yaml` to version `0.2.0`, set the real source tarball URL/sha256, fix the PyPI `home` link, add yourself to `recipe-maintainers`, and align dependencies with `pyproject.toml`.
- Ensure `environment.yml` pins dev extras (or document that it installs the published wheel) and note any manual steps in the README.
- Add a lightweight CI job (or document a local workflow) for `make acceptance`; if it remains local-only, mention that in `docs/guides/CHANGELOG.md` or contributing docs.
- Confirm GitHub workflows (`ci.yml`, `docs.yml`, `build-and-deploy.yml`, `release.yml`) stay green after the recipe/docs changes; adjust caching/install steps as needed.
- Consider publishing coverage to Codecov only when the token is present—make the step conditional.

Validation:
- `conda mambabuild recipe` (or `conda-build recipe`) to ensure the recipe is valid.
- `pytest -q`
- Dry-run relevant GitHub workflow steps locally if feasible (`act` optional).

---

# Prompt 4 – API & Infrastructure Hardening

Context: `/gpfs/commons/home/jameslee/scPerturb-CMap`

Goals:
- Introduce a config module or utilities that read API settings (LINCS path, model path, cache TTL, CORS origins, metrics backend) from environment variables with sane defaults; wire these into `scripts/api/main.py` and document them in `deployment/docker/README.md` and `docs/deployment/CLOUD_DEPLOYMENT.md`.
- Implement a readiness probe in the FastAPI app that verifies the LINCS cache/model is available (and optionally tests Redis/Postgres when enabled) before returning `ready`.
- Add graceful handling for oversized requests, request timeouts, and clearer HTTP error responses.
- Restrict CORS defaults to configured origins, falling back to `*` only in development.
- Ensure the Prometheus endpoint and background metric tasks work under the new config.
- Update Helm values/docs to expose the new environment variables and note security expectations (API keys, rate limiting left for future).

Validation:
- Start the API locally (`uvicorn scripts.api.main:app`) with mocked env vars and hit `/health`, `/ready`, `/api/score`.
- Run unit tests touching the new config paths (add them under `tests/test_api.py` if needed).

---

# Prompt 5 – Final Polish & Validation

Context: `/gpfs/commons/home/jameslee/scPerturb-CMap`

Goals:
- Add a licensing/data-use note in `README.md` or `docs/guides/DATA_NOTES.md` clarifying LINCS usage, attribution, and redistribution guidance.
- Re-run the Streamlit UI, exercise export/bookmark flows, and capture screenshots if any UI changed; update `docs/assets/` if needed.
- Ensure “make acceptance” succeeds on a clean checkout and document expected runtime/output in `docs/quickstart.md` or contributing guides.
- Review `examples/out/` artifacts; ensure tracked files represent intentional demo outputs and add `.gitignore` entries for transient artifacts if necessary.
- Perform a final spelling/grammar pass across README and docs (use `codespell` or similar) and record fixes in the changelog.

Validation:
- `make acceptance`
- `streamlit run src/scperturb_cmap/ui/app.py` (manual verification)
- `codespell --ignore-words-list=moa` (or similar command)
- Update `docs/guides/CHANGELOG.md` with a concise “Ready for 0.2.0 GA” entry summarizing the above work.

