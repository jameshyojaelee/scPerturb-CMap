# scPerturb-CMap

scPerturb-CMap is a lightweight Python package scaffold for ML utilities and tools related to perturbational profiling and drug discovery workflows. It follows a modern Python project layout with a `src/` structure, includes linting with Ruff, testing with PyTest, and a simple device auto-selection helper supporting CPU, CUDA, and Apple MPS.

## Features
- Python 3.10+
- `src/` package layout
- Installable via `pyproject.toml`
- Lint with `ruff`, test with `pytest`
- Simple device helper: auto-selects `cuda`, `mps`, or `cpu`

## Quickstart

```bash
# Setup (creates venv and installs dev extras)
make setup

# Lint and test
make lint
make test
```

## Makefile targets
- `make setup`: Create venv (if missing) and install package with dev deps
- `make lint`: Run ruff checks on `src` and `tests`
- `make test`: Run pytest
- `make ui`: Placeholder for launching Streamlit app
- `make demo`: Print detected device using the helper

## CLI
After installation, a CLI command `scperturb-cmap` is available.

## Device helper
The device helper returns one of `cuda`, `mps`, or `cpu` based on availability. If PyTorch is not installed, it will default to `cpu` without failing.

```python
from scperturb_cmap.utils.device import get_device
print(get_device())  # "cuda" | "mps" | "cpu"
```

## License
MIT License. See `LICENSE` for details.
