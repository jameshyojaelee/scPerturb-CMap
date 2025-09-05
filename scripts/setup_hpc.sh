#!/bin/bash
set -euo pipefail

# HPC setup script for scPerturb-CMap
# Usage:
#   SCPC_BASE=/gpfs/commons/home/$USER/scPerturb-CMap make hpc-setup
# If SCPC_BASE is not set, defaults to the repository root.

# Resolve repository root (directory above this script)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

# Where to create data/output dirs (can be outside the repo)
DATA_BASE="${SCPC_BASE:-$repo_root}"

echo "Setting up scPerturb-CMap HPC environment..."
echo "Repo root:  $repo_root"
echo "Data base:  $DATA_BASE"

# Create directory structure
echo "Creating directory structure..."
mkdir -p "$DATA_BASE"/{data/raw,data/lincs,data/sc,artifacts,examples/out}

# Show directory tree
echo "Created directories:"
echo "$DATA_BASE/data/raw/"
echo "$DATA_BASE/data/lincs/"
echo "$DATA_BASE/data/sc/"
echo "$DATA_BASE/artifacts/"
echo "$DATA_BASE/examples/out/"

# Check for conda/mamba and create environment
if command -v mamba &> /dev/null; then
    echo "Using mamba..."
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    echo "Using conda..."
    CONDA_CMD="conda"
else
    echo "Neither conda nor mamba found. Continuing with current Python environment..."
fi

# Check if environment already exists (only if conda is available)
if [[ -n "${CONDA_CMD:-}" ]]; then
    if $CONDA_CMD env list | grep -q "^scpc "; then
        echo "Environment 'scpc' already exists. Activating..."
        eval "$($CONDA_CMD shell.bash hook)"
        conda activate scpc
    else
        echo "Creating conda environment 'scpc' with Python 3.10..."
        $CONDA_CMD create -n scpc python=3.10 -y
        eval "$($CONDA_CMD shell.bash hook)"
        conda activate scpc
    fi
fi

# Install the package from the repo root
echo "Installing scPerturb-CMap with dev dependencies..."
cd "$repo_root"
pip install -e ".[dev]"

# Test device detection
echo "Testing device detection..."
python -c "from scperturb_cmap.utils.device import get_device; print(f'PyTorch device: {get_device()}')"

echo "HPC setup complete!"
