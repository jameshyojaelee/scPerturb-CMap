
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

