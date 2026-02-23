#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"

conda create -n nanofm python=3.10 -y
conda activate nanofm

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

pip install --upgrade pip
pip install torch torchvision wandb einops datasets transformers diffusers safetensors torchmetrics torch-fidelity huggingface-hub accelerate
pip install -e .
pip install git+https://github.com/NVIDIA/Cosmos-Tokenizer.git --no-dependencies
python -m ipykernel install --user --name nanofm --display-name "nano4M kernel (nanofm)"

echo "torchrun location: $(which torchrun)"
echo "Setup complete."