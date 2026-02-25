# SCITAS Tutorial

- [SCITAS Tutorial](#scitas-tutorial)
- [What is SCITAS](#what-is-scitas)
- [Terminology](#terminology)
- [How to create an account](#how-to-create-an-account)
- [How to access to the cluster](#how-to-access-to-the-cluster)
  - [Login](#login)
  - [Volumes](#volumes)
    - [Home](#home)
    - [Scratch](#scratch)
  - [Upload data](#upload-data)
- [Running a job on the cluster](#running-a-job-on-the-cluster)
  - [Submitting a job](#submitting-a-job)
  - [How to run interactive jobs on SCITAS](#how-to-run-interactive-jobs-on-scitas)
  - [GPU resources](#gpu-resources)
  - [Using Apptainer/Singularity to containerize your job](#using-apptainersingularity-to-containerize-your-job)
- [House keeping](#house-keeping)

---
# What is SCITAS
In CS-503 Visual Intelligence: Machines and Minds, we are going to use the Scientific IT and Application Support (SCITAS) [[link](https://scitas-doc.epfl.ch/)] cluster for computation. It provides scientific computing resources and High Performance Computing (HPC) services to everyone at EPFL. Currently, it contains four clusters: Helvetios, Izar GPU, Izar GPU 4x, and Jed [[link](https://scitas-doc.epfl.ch/supercomputers/overview/)]. We will mainly use [Izar](https://scitas-doc.epfl.ch/supercomputers/izar/) for our GPU computation.

# Terminology
When you see ***locally*** or ***on your own computer***, it means the command should be executed on your own computer, not the cluster.

When you see ***remotely*** or ***on the clusters***, it means the command should be executed on the clusters (after logging in), not your own computer.

We use `<username>` to denote your username on the clusters. Please replace it with your actual username when you execute the commands.

# How to create an account

No need to do it yourself. Once you have enrolled yourself in the course, an account will be created automatically for you that uses your EPFL Gaspar credentials. Your account will be associated with the CS503 project to use reserved GPUs or acquire high priority in job queues. Please reach out to the teaching staff via email (vimm-ta@groupes.epfl.ch) in case you face any difficulty in using your account.

# How to access to the cluster
To connect to the clusters, you have to be inside the EPFL network or [establish a VPN connection](https://www.epfl.ch/campus/services/en/it-services/network-services/remote-intranet-access/vpn-clients-available/) [[link](https://scitas-doc.epfl.ch/user-guide/using-clusters/connecting-to-the-clusters/)].

## Login
You can access the clusters by using `ssh` on your own computer. The command is:
```bash
ssh -X <username>@izar.epfl.ch
```

## Volumes
The volumes mentioned below are the folders existing on the clusters.

### Home
You have 100 GB quota in `/home/<username>` for storing important files such as codes. The files here are backed up every night and the storage is kept permanantly.

### Scratch
`/scratch/<username>` is used to store large datasets, checkpoints, etc. Files here are **NOT backed up** and the files **older than two week can get deleted**. Therefore, please only store files that you can afford to lose and reproduce easily here.

## Upload data
Sometimes, you need to upload data to the clusters or download data from the clusters [[link](https://scitas-doc.epfl.ch/user-guide/data-management/transferring-data/)]. On your own computer, you can use `rsync` [[link](https://scitas-doc.epfl.ch/user-guide/data-management/transferring-data/#using-rsync)] (recommended) and `scp` [[link](https://scitas-doc.epfl.ch/user-guide/data-management/transferring-data/#using-scp)] if you prefer command line tools. If you prefer GUI applications, you can also use [WinSCP](https://winscp.net/eng/index.php) on Windows or [FileZilla](https://filezilla-project.org/) on MacOS and Linux locally.

# Running a job on the cluster
The SCITAS clusters use [SLURM](https://slurm.schedmd.com/documentation.html) to manage jobs submitted. It is a commonly used job scheduling system for HPC clusters. You can find almost all the information you want to know about SLURM on its [official documentation](https://slurm.schedmd.com/documentation.html) or by searching on Google.

## Submitting a job
For this project, use the same workflow as in `NanoFM_Homeworks/README.md`.

1. Install dependencies (once, on the cluster):
   ```bash
   bash setup_env.sh
   ```

2. Run training in one of two ways:

   - Option 1: Interactive (`srun`) for debugging:
     ```bash
     srun -t 120 -A cs-503 --qos=cs-503 --gres=gpu:2 --mem=16G --pty bash
     ```
     Then on the compute node:
     ```bash
     conda activate nanofm
     wandb login
     OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_training.py --config cfgs/nanoGPT/tinystories_d8w512.yaml
     ```
     For 1 GPU, use `--gres=gpu:1` and `--nproc_per_node=1`.

   - Option 2: Batch job (`sbatch`) for longer runs:
     ```bash
     sbatch submit_job.sh <config_file> <your_wandb_key> <num_gpus>
     ```
     Example:
     ```bash
     sbatch submit_job.sh cfgs/nanoGPT/tinystories_d8w512.yaml abcdef1234567890 2
     ```

3. For multi-node training (Part 3), submit:
   ```bash
   sbatch submit_job_multi_node_scitas.sh <config_file> <your_wandb_key>
   ```
   Example:
   ```bash
   sbatch submit_job_multi_node_scitas.sh cfgs/nano4M/multiclevr_d6-6w512.yaml abcdef1234567890
   ```

4. After submission, SLURM prints `Submitted batch job <job-id>`.
5. Cancel a job with:
   ```bash
   scancel <job-id>
   ```
6. Check your jobs with:
   ```bash
   squeue -u <username>
   ```

## How to run interactive jobs on SCITAS

Sometimes you want to **interactively debug or run notebooks** on a GPU node (e.g., from VS Code or a local browser). This section shows two common workflows.

### Option 1: Remote-SSH into a SLURM session created with `srun`

#### Create a SLURM session

First create an interactive SLURM job using `srun`, for example:

```bash
srun -t 120 -A cs-503 --qos=cs-503 --gres=gpu:2 --mem=16G --pty bash
```

You should now be in the interactive SLURM job. Check `hostname` to confirm. It should return the name of the GPU node within the SCITAS Izar cluster that you are currently on, e.g.:

```bash
i30
```

You can now directly run your Python scripts in the CLI using commands such as:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_training.py --config cfgs/nanoGPT/tinystories_d8w512.yaml
```

But to run a Jupyter notebook *interactively* through something like VS Code, you need to be able to open a **Remote-SSH session** to the SLURM job, *not* just `izar`. The steps below show you how to achieve this.

#### Copy your local public key into the GPU node

On your **local machine**, first check that you have an SSH key (commands shown for macOS; Linux is similar):

```bash
ls -l ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub
```

If it does not exist, create one:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
```

Copy your local (laptop's) public key onto the GPU node (replace `<username>` and `i30`):

```bash
cat ~/.ssh/id_ed25519.pub | ssh <username>@izar.epfl.ch 'ssh <username>@i30 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"'
```

Make sure to replace `<username>` with your actual username that you use to access `izar`. In addition, replace `i30` with the name of the GPU node that you are on (returned by `hostname` as explained above).

This command copies your public key, which is needed for SSH connections, onto your GPU node *through* `izar`.

#### Check that you can SSH into the GPU node

From your **local machine**:

```bash
ssh -J <username>@izar.epfl.ch <username>@i30 hostname
```

This should print the name of your GPU node again. Success!

#### Edit `~/.ssh/config`

Now add the following entry to `~/.ssh/config` (just copy–paste the following snippet into the file), replacing `<username>` and `i30`:

```bash
Host izar
  HostName izar.epfl.ch
  User <username>
  ForwardX11 yes

Host izar-gpu
  HostName i30
  User <username>
  ProxyJump izar
```

Now, you should be able to Remote-SSH into the SLURM job (GPU session), not just `izar`, using:

- `ssh izar-gpu` on your terminal, or  
- VS Code's `Remote-SSH: Connect to Host` command and selecting `izar-gpu`.

### Option 2: Run a Jupyter notebook on a GPU node and port-forward

This option uses a long-lived terminal session (e.g. with `tmux`) plus SSH port forwarding to access Jupyter from your local browser.

#### 1. Start a persistent terminal on Izar

On the cluster (logged into `izar`):

```bash
module load tmux
tmux new -s jupyter
```

You are now inside a `tmux` session named `jupyter`.

#### 2. Start an interactive GPU job inside `tmux`

Inside `tmux`, request a GPU node:

```bash
srun -t 120 -A cs-503 --qos=cs-503 --gres=gpu:1 --mem=16G --pty bash
```

You are now on a GPU node (again, check with `hostname`).

#### 3. Activate the CS503 environment

If you used the course setup script from `NanoFM_Homeworks`:

```bash
conda activate nanofm
```

Alternatively, if you rely on your own environment, make sure it contains all needed packages and that your conda-related directories appear early in `PATH`.

#### 4. Start the Jupyter server on the GPU node

From the GPU node, in the CS503 repository directory (adapt the path as needed):

```bash
cd /home/<username>/CS503/2026-spring/NanoFM_Homeworks
jupyter lab --no-browser --port=8888 --ip=$(hostname -i)
```

The output should contain a line similar to:

```text
http://10.91.27.4:8888/lab?token=<token>
```

Where:

- `<ip-address>` is `10.91.27.4` in this example.  
- `<token>` is the long token string at the end of the URL.

#### 5. Forward the port to your local machine

On your **local machine**, execute (replace `<ip-address>` and `<username>`):

```bash
ssh -L 8888:<ip-address>:8888 -l <username> izar.epfl.ch -f -N
```

This forwards the remote Jupyter port to `localhost:8888` on your laptop.

#### 6. Open Jupyter in your browser

On your **local machine**, open a web browser and enter:

```text
http://127.0.0.1:8888/lab?token=<token>
```

Replace `<token>` with the token printed in the Jupyter server output. You should now see your Jupyter Lab session running on the GPU node.

You can keep the job running in the background by detaching from `tmux` with `Ctrl+b d`, and later reattach with:

```bash
tmux attach -t jupyter
```

## GPU resources
In the CS-503 course, we have reserved GPUs that are only accessible to CS-503 students. For SCITAS commands and job scripts, use the CS-503 account and QoS.

```bash
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
```



# House keeping

Please be considerate to the other students when using the clusters.

We do not restrict the number of GPUs each group can use. You can use all available resources, but other groups will not be able to use it, i.e., no preemption.
