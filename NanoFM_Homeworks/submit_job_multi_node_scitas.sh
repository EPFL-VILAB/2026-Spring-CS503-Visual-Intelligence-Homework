#!/bin/bash
#SBATCH --job-name=multi_node_job
#SBATCH --time=12:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=multi_node_job.out
#SBATCH --error=multi_node_job.err

# === Accept arguments ===
CONFIG_FILE=$1       # First argument
WANDB_KEY=$2        # Second argument


# === Initialization ===
set -x
cat $0
export MASTER_PORT=25678
export MASTER_ADDR=$(hostname)
export WANDB_API_KEY=$WANDB_KEY
export NCCL_DEBUG=INFO

# === Run main script ===
srun bash -c "
  TORCHRUN_ARGS=\"--node-rank=\${SLURM_PROCID} \
     --master-addr=\${MASTER_ADDR} \
     --master-port=\${MASTER_PORT} \
     --nnodes=\${SLURM_NNODES} \
     --nproc-per-node=2\"

  echo \${SLURM_PROCID}
  echo \${TORCHRUN_ARGS}
  echo \${SLURMD_NODENAME}

  torchrun \${TORCHRUN_ARGS} run_training.py \
    --config $CONFIG_FILE
"

