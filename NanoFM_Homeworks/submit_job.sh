#!/bin/bash
#SBATCH --job-name=you_job         # Change as needed
#SBATCH --time=02:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2                    # Request 2 GPUs
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # Adjust CPU allocation if needed
#SBATCH --output=interactive_job.out    # Output log file
#SBATCH --error=interactive_job.err     # Error log file

CONFIG_FILE=$1
WANDB=$2
NUM_GPUS=$3

conda activate nanofm
export WANDB_API_KEY=$WANDB && OMP_NUM_THREADS=1 torchrun --nproc_per_node=$NUM_GPUS run_training.py --config $CONFIG_FILE
