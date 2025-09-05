Contributing to scPerturb-CMap

Thanks for your interest in contributing! This project aims to provide a clean, testable scaffold for single‑cell connectivity mapping with both baseline and learned (metric) methods.

Development workflow
- Fork and branch: Create a feature branch off `main`.
- Setup env: `make setup` (creates `.venv` and installs `-e .[dev]`).
- Code style: Keep changes minimal and focused. Follow existing file layout and naming.
- Lint: `make lint` (Ruff, line length 100, target Python 3.10+).
- Test: `make test` (pytest; add unit tests for new code). Prefer fast, deterministic tests.
- Demo checks: `make demo` to generate demo data and verify scoring works; `make ui` to sanity‑check the app.
- Commit messages: Use concise, descriptive messages (e.g., `feat(ui): add score export`).
- Pull request: Link issues when applicable, summarize changes and any trade‑offs.

Code guidelines
- Keep public APIs stable and typed. Use Pydantic v2 models for configs/schemas.
- Prefer pure functions & small modules; add docstrings where behavior isn’t obvious.
- Avoid global state and heavy imports at module top‑level (keep import‑time light).
- Follow tests-first when possible; include edge cases and negative tests.
- Keep optional heavy deps and long‑running examples out of unit tests.

Local useful targets
- `make setup` – create venv and install dev deps
- `make lint` – run Ruff on `src` and `tests`
- `make test` – run pytest
- `make demo` – synthesize demo data and run baseline scoring
- `make ui` – launch the Streamlit demo app
- `make train` – run a small Hydra training job
- `make evaluate` – evaluate the checkpoint in `artifacts/best.pt`
- `make acceptance` – run acceptance checks on demo workload

Security and data
- Do not commit secrets, real patient data, or licensed datasets.
- Synthetic demo data is generated under `examples/data/`.

License
- MIT. By contributing, you agree your contributions are provided under the project license.

