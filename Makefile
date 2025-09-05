PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff

.PHONY: setup lint test ui demo train evaluate acceptance hpc-setup hpc-ui

setup:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e .[dev]

lint:
	$(RUFF) check src tests

test:
	$(PYTEST) -q

ui:
	python -m streamlit run src/scperturb_cmap/ui/app.py

train:
	$(PY) -m scperturb_cmap.models.train hydra.run.dir=.

demo:
	$(PY) scripts/make_demo_h5ad.py --output examples/data/demo.h5ad
	$(PY) scripts/print_demo_stats.py
	@mkdir -p examples/out
	@echo '{"genes":["G1","G2","G10"],"weights":[1.0,1.0,-1.0],"metadata":{}}' > examples/out/target.json
	$(PY) -m scperturb_cmap.cli score \
		--target-json examples/out/target.json \
		--library examples/data/lincs_demo.parquet \
		--method baseline \
		--top-k 50 \
		--output examples/out/results.parquet

evaluate:
	$(PY) -c "from scperturb_cmap.models.evaluate import evaluate_checkpoint; import json; print(json.dumps(evaluate_checkpoint('artifacts/best.pt')))"

acceptance:
	$(PY) scripts/check_acceptance.py

hpc-setup:
	chmod +x scripts/setup_hpc.sh
	./scripts/setup_hpc.sh

hpc-ui:
	chmod +x scripts/ui_hpc.sh
	./scripts/ui_hpc.sh
