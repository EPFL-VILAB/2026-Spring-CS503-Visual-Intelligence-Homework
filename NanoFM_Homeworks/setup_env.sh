#!/usr/bin/env bash
set -euo pipefail

# Create virtual environment using uv if it doesn't already exist
if [ ! -d "nanofm" ]; then
  uv venv nanofm
fi

# Activate the virtual environment
source .venv/bin/activate

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# Install dependencies using uv
uv pip install --upgrade pip
uv pip install torch torchvision wandb einops datasets transformers diffusers safetensors torchmetrics torch-fidelity huggingface-hub accelerate
uv pip install -e .
uv pip install git+https://github.com/NVIDIA/Cosmos-Tokenizer.git --no-deps
python -m ipykernel install --user --name nanofm --display-name "nano4M kernel (nanofm)"

echo "torchrun location: $(which torchrun)"
echo "Setup complete."