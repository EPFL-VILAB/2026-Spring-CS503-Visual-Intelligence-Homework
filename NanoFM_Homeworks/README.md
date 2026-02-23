# CS503 – Foundation Model Homework (nanoGPT, nanoMaskGIT, nano4M)

Welcome to the foundation model exercises! This homework consists of five notebook assignments, all implemented in this `NanoFM_Homeworks` directory:  

1. **Part 1 – nanoGPT (autoregressive Transformer)**  
   - Implement the necessary building blocks to construct an autoregressive Transformer (GPT-style) for text and image token generation.
   - Main notebook: `notebooks/CS503_FM_part1_nanoGPT.ipynb`.  
   - Main code files: `nanofm/models/gpt.py`, `nanofm/modeling/transformer_layers.py`.

2. **Part 2 – nanoMaskGIT (masked generative model)**  
   - Build a masked token model in the style of MaskGIT for text and image generation.
   - Main notebook: `notebooks/CS503_FM_part2_nanoMaskGIT.ipynb`.  
   - Main code file: `nanofm/models/maskgit.py`.

3. **Part 3 – nano4M (simple multimodal model)**  
   - Build a simplified 4M-like multimodal model for text–image generation.
   - Main notebook: `notebooks/CS503_FM_part3_nano4M.ipynb`.  
   - Main code file: `nanofm/models/fourm.py`.

4. **(Coming soon) nanoVLM & nanoFlowMatching**  
   - Later in the course, you will work with a small vision–language model (**nanoVLM**) and a flow-matching model (**nanoFlowMatching**).  
   - The corresponding notebooks, configs, and code skeletons will be **released later**; follow the course announcements and this README for updates.

### Instructions

The detailed instructions for each of these three parts are provided in the corresponding notebooks under `./notebooks/`. They:

- Introduce the problem statement for each part.  
- Explain which functions in the codebase need to be implemented or completed.  
- Guide you through training, evaluation, and qualitative inspection of the models.  

You are expected to run all required cells in the notebooks, answer any questions they contain, and save the outputs before submission.

## 1 Setup

### 1.1 Dependencies and environment

The notebook should be run on one GPU, while the actual training requires 1-2 GPUs depending on the config.

The required packages for training are specified in `pyproject.toml`.
We provide a convenience script, `setup_env.sh`, which creates a `nanofm` environment and Jupyter kernel, and installs the requirements.

Run:

```bash
bash setup_env.sh
```

After setup, activate the environment with:

```bash
conda activate nanofm
```

In Jupyter, select the `nano4M kernel (nanofm)` kernel when running notebook cells.

### 1.2 Codebase overview

This nano4M codebase is a heavily simplified version of the original 4M codebase. We use it to implement nano versions of GPT for autoregressive text and image generation, MaskGIT for masked text and image generation, and 4M for multimodal generation.

The codebase is structured as follows:

- `cfgs/`: Configs specifying what model to train, on what data, and for how long.
- `nanofm/`: Main package directory.
  - `data/`: Data loaders for text, vision, and multimodal datasets.
  - `modeling/`: Shared modeling utilities such as Transformer layer definitions.
  - `models/`: Concrete model implementations including forward, loss, and generation logic.
  - `utils/`: Helper utilities for training, checkpointing, and related workflows.
- `notebooks/`: Instructions for parts 1-3. These notebooks are part of what you submit.
- `run_training.py`: Main training and evaluation entry point.

### 1.3 Weights & Biases setup

When training models, you should log training/validation metrics and training-health statistics.
For example, you will track train and validation losses (for convergence and overfitting checks) and gradient norms (for stability checks).

We use Weights & Biases (W&B), which also logs system metrics such as GPU utilization and memory usage.

1. Create a W&B account at [wandb.ai](https://wandb.ai/) if you do not already have one.
2. Open [wandb.ai/settings](https://wandb.ai/settings) and copy your API key (or create a new one).
3. Log in from the terminal:
   ```bash
   wandb login <KEY>
   ```
4. Alternatively, set the key as an environment variable:
   ```bash
   export WANDB_API_KEY=<KEY>
   ```
5. On every new compute node, make sure you log in before training.
6. Set `wandb_entity` in your config files to your W&B username.

## 2 Running training jobs

### Main training script `run_training.py`

You can run the training job in one of two ways:

1. **Interactively using `srun`** – great for debugging.
2. **Using a SLURM batch script** – better for running longer jobs.

> **Before you begin**:  
> Make sure to have your Weights & Biases (W&B) account and obtain your W&B API key.  
> Follow **Section 1.3 (Weights & Biases setup)** in this README.

---

#### Option 1: Run Interactively via `srun`

Start an interactive session on a compute node (eg, 2 GPUs case):

```bash
srun -t 120 -A cs-503 --qos=cs-503 --gres=gpu:2 --mem=16G --pty bash
```
Then, on the compute node:

```bash
conda activate nanofm
wandb login
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_training.py --config cfgs/nanoGPT/tinystories_d8w512.yaml
```
> **Note:**  
> To run the job on **one GPU**, make sure to:
> - Adjust the `--gres=gpu:1` option in the `srun` command, and  
> - Set `--nproc_per_node=1` in the `torchrun` command.

#### Option 2: Submit as a Batch Job via SLURM
You can use the provided submit_job.sh script to request GPUs and launch training.

Run:
```bash
sbatch submit_job.sh <config_file> <your_wandb_key> <num_gpus>
```
Replace the placeholders as follows:

- <config_file> — Path to your YAML config file

- <your_wandb_key> — Your W&B API key

- <num_gpus> — Set to 1 or 2 depending on your setup

Example Usage:
```bash
sbatch submit_job.sh cfgs/nanoGPT/tinystories_d8w512.yaml abcdef1234567890 2
```
#### Multi-node Training: Submit as a Batch Job via SLURM

For the third part of nano4M, we will scale up the training compute by utilizing 4 GPUs. To do this, we will train models using a multi-node GPU setup on the IZAR Cluster.

Most commands remain the same as before, and we will use a specific multi-node training sbatch script.

Run:
```bash
sbatch submit_job_multi_node_scitas.sh <config_file> <your_wandb_key>
```

Replace the placeholders as follows:

- <config_file> — Path to your YAML config file

- <your_wandb_key> — Your W&B API key


Example Usage:
```bash
sbatch submit_job_multi_node_scitas.sh cfgs/nano4M/multiclevr_d6-6w512.yaml abcdef1234567890
```
