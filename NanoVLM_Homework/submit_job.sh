#!/bin/bash
#SBATCH --job-name=nanovlm_train
#SBATCH --time=05:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=nanovlm_train_%j.out
#SBATCH --error=nanovlm_train_%j.err

# Batch-submit NanoVLM training on SCITAS (Izar).
#
# Usage (on the cluster; run from NanoVLM_Homework, or: sbatch --chdir=/path/to/NanoVLM_Homework ...):
#   cd /path/to/NanoVLM_Homework && sbatch submit_job.sh <wandb_api_key> <huggingface_hub_token> [num_gpus]
#
# Example (2 GPUs; keys taken from your environment — do not commit real secrets):
#   sbatch submit_job.sh "$WANDB_API_KEY" "$HUGGINGFACE_HUB_TOKEN" 2


cd "${SLURM_SUBMIT_DIR:-.}"

WANDB_KEY="${1:?Usage: sbatch submit_job.sh <wandb_api_key> <huggingface_hub_token> [num_gpus]}"
HF_TOKEN="${2:?Usage: sbatch submit_job.sh <wandb_api_key> <huggingface_hub_token> [num_gpus]}"
NUM_GPUS="${3:-2}"

# Hugging Face cache / Hub data root on the course shared volume (adjust if your path differs).
export HF_HOME=/work/com-304/com-304-nanovlm-dataset/

# No Hub HTTP (avoids timeouts on compute nodes). Requires a complete cache under HF_HOME; comment out if downloads are needed.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export PYTHONUNBUFFERED=1
export WANDB_API_KEY="$WANDB_KEY"

# Non-interactive batch shells often lack conda hook; common miniconda layout on Izar:
# shellcheck disable=SC1091
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook 2>/dev/null)" || true
fi
conda activate nanofm

OMP_NUM_THREADS=1 torchrun --nproc_per_node="${NUM_GPUS}" train.py
