# Contributing to scPerturb-CMap

Thanks for your interest in contributing! This project aims to provide a clean,
testable scaffold for single-cell connectivity mapping with both baseline and
learned (metric) methods.

## Development Setup

- Fork and branch: create a feature branch off `main`.
- Set up the environment: `make setup` (creates `.venv` and installs `-e .[dev]`).
- Keep changes minimal and focused. Follow existing file layout and naming.
- Lint: `make lint` (Ruff, line length 100, target Python 3.10+).
- Test: `make test` (pytest; add unit tests for new code). Prefer fast,
  deterministic tests.
- Demo checks: `make demo` to generate demo data and verify scoring works;
  `make ui` to sanity-check the app.
- Commit messages: use concise, descriptive messages (e.g.,
  `feat(ui): add score export`).
- Pull request: link issues when applicable, summarise changes and any
  trade-offs.

## Code Style

- Keep public APIs stable and typed. Use Pydantic v2 models for configs and
  schemas.
- Prefer pure functions and small modules; add docstrings where behaviour isn’t
  obvious.
- Avoid global state and heavy imports at module top level (keep import time
  light).
- Follow tests-first when possible; include edge cases and negative tests.
- Keep optional heavy dependencies and long-running examples out of unit tests.

## Useful Make Targets

- `make setup` – create venv and install dev dependencies
- `make lint` – run Ruff on `src/` and `tests/`
- `make test` – run pytest
- `make demo` – synthesise demo data and run baseline scoring
- `make ui` – launch the Streamlit demo app
- `make train` – run a small Hydra training job
- `make evaluate` – evaluate `workspace/artifacts/best.pt`
- `make acceptance` – run acceptance checks on the demo workload (local-only; not part of CI)

## Security and Data Handling

- Do not commit secrets, real patient data, or licensed datasets.
- Synthetic demo data is generated under `examples/data/`.

## License

MIT. By contributing, you agree your contributions are provided under the
project license.
