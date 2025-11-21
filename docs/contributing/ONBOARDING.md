# Contributor Onboarding

Welcome! This short guide gets you from clone to a passing test run.

## Setup

```bash
git clone https://github.com/jameslee/scPerturb-CMap.git
cd scPerturb-CMap
make setup            # creates .venv/ and installs dev dependencies
make demo             # builds the demo LINCS assets and synthetic target
make test             # quick test suite
```

Optional: install pre-commit so `ruff` and `pytest -q` run on staged files:

```bash
pip install pre-commit
pre-commit install
```

## Working style

- Keep changes small and well tested (`make lint`, `make test`).
- Use the issue/PR templates under `.github/` to capture context.
- Follow the [Code of Conduct](../../CODE_OF_CONDUCT.md) when collaborating.

For deeper guidance, see `docs/contributing/CONTRIBUTING.md`.
