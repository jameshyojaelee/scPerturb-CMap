PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff

.PHONY: setup lint test ui demo

setup:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e .[dev]

lint:
	$(RUFF) check src tests

test:
	$(PYTEST) -q

ui:
	python -m streamlit run $$(python -c 'import scperturb_cmap.ui.app,inspect; print(scperturb_cmap.ui.app.__file__)')

demo:
	$(PY) -c "from scperturb_cmap.utils.device import get_device; print('Device:', get_device())"
 