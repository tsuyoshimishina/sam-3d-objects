#!/bin/bash
# SAM 3D Objects - Environment Setup Script
#
# Usage:
#   cd /path/to/sam-3d-objects
#   ./setup.sh
#
# Prerequisites:
#   - Linux 64-bit architecture
#   - NVIDIA GPU with at least 32GB VRAM
#   - HuggingFace account with:
#     - Access granted to facebook/sam-3d-objects model
#     - Access token with Read permission (https://huggingface.co/settings/tokens)

set -eo pipefail

MINIFORGE_DIR="$HOME/miniforge3"
ENV_NAME="sam3d-objects"

echo "[Step 1] Installing Miniforge (Mamba)..."

if command -v conda &> /dev/null; then
    echo "Error: conda is already installed. This script requires a fresh Miniforge installation."
    echo "Please remove existing conda/miniconda/anaconda and re-run this script."
    exit 1
fi

if [ ! -f "$MINIFORGE_DIR/bin/mamba" ]; then
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash "Miniforge3-$(uname)-$(uname -m).sh" -b -p "$MINIFORGE_DIR"
    rm -f "Miniforge3-$(uname)-$(uname -m).sh"
    "$MINIFORGE_DIR/bin/conda" init bash
    echo "Miniforge installed. Please restart your shell and re-run this script."
    exit 0
fi

source "$MINIFORGE_DIR/etc/profile.d/conda.sh"

echo "[Step 2] Creating mamba environment..."

if ! mamba env list | grep -q "^$ENV_NAME "; then
    mamba env create -f environments/default.yml
fi

conda activate "$ENV_NAME"

echo "[Step 3] Installing Python dependencies..."

export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'
pip install -e '.[p3d]'

export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

echo "[Step 4] Applying patches..."

./patching/hydra

echo "[Step 5] Downloading checkpoints..."

if [ ! -d "checkpoints/hf" ] || [ -z "$(ls -A checkpoints/hf 2>/dev/null)" ]; then
    pip install 'huggingface-hub[cli]<1.0'
    hf auth login
    # Download to temp dir and extract checkpoints/ subdirectory,
    # to avoid nested path like checkpoints/hf/checkpoints/pipeline.yaml
    hf download --repo-type model --local-dir checkpoints/hf-download --max-workers 1 facebook/sam-3d-objects
    mv checkpoints/hf-download/checkpoints checkpoints/hf
    rm -rf checkpoints/hf-download
fi

# Done
echo ""
echo "Setup complete!"
echo "  conda activate $ENV_NAME"
echo "  python demo.py"
