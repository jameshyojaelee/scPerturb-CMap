#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .[dev]
make acceptance
