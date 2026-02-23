# CS503 2026 Spring – Homework Repository

Welcome to the CS503 homework repository. This repository contains all homework and tutorial materials used throughout the course.

## Repository structure

1. [Transformer_Homework](/Transformer_Homework)
   - Homework 1: introduction to transformer architectures for sequence modeling.

2. [NanoFM_Homeworks](/NanoFM_Homeworks)
   - Homework 2: `nanoGPT` and `nanoMaskGIT` — small-scale implementations of GPT-style language modeling and MaskGIT-style image token modeling.
   - Homework 3: `nano4M` — a small-scale implementation of 4M-style multi-modal model.
   - Homework 4: `nanoVLM` and `nanoFlowMatching` (to be released) — vision-language modeling and flow-matching based generative modeling.
   - This folder also includes environment setup scripts, cluster submission scripts, and training utilities shared across the NanoFM homeworks.

In addition, we provide several tutorials:

1. [Cluster_Tutorial](/Cluster_Tutorial): overview of the two main compute resources used in this course (SCITAS and GNOTO) and how to run jobs on them. We **recommend** running the Transformer homework on GNOTO and the NanoFM homeworks on SCITAS.
- [SCITAS_Tutorial](/Cluster_Tutorial/SCITAS_Tutorial.md): detailed instructions for running interactive and batch jobs on SCITAS (Izar).
- [GNOTO_Tutorial](/Cluster_Tutorial/gnoto_tutorial.md): detailed guide for running in a JupyterLab-based environment (GNOTO).
2. [PyTorch_Tutorial](/PyTorch_Tutorial): a brief PyTorch refresher for those not familiar with PyTorch (*however, we do not recommend taking this course if this is your first time using PyTorch*). The tutorial will not be graded.

**Important note on compute resources:** GPU resources are limited and shared across the class. **Do not wait until the last day** to start or finish your homework. If you cannot obtain GPUs due to last-minute congestion, this will **not** be accepted as a valid reason for late submission or deadline extension.

## Submission details
We provide an overview of the submission requirements, score distribution, and deadlines below. Please also carefully follow the detailed instructions in each notebook and any announcements on Moodle for the most up-to-date requirements.

1. **Transformer Homework** (5%) — **due 8 March, 23:59**
   - Submit:
     - [`CS503_Transformer_Homework.ipynb`](/Transformer_Homework/CS503_Transformer_Homework.ipynb) with all required cells completed and outputs saved.

2. **nanoGPT (5%) & nanoMaskGIT (10%)** — **due 22 March, 23:59**
   - Submit:
     - [`CS503_FM_part1_nanoGPT.ipynb`](/NanoFM_Homeworks/notebooks/CS503_FM_part1_nanoGPT.ipynb) with all required cells completed and outputs saved.
     - [`gpt.py`](/NanoFM_Homeworks/nanofm/models/gpt.py) with all required sections implemented.
     - [`transformer_layers.py`](/NanoFM_Homeworks/nanofm/modeling/transformer_layers.py) with task-specific functions implemented.
     - [`CS503_FM_part2_nanoMaskGIT.ipynb`](/NanoFM_Homeworks/notebooks/CS503_FM_part2_nanoMaskGIT.ipynb) with all required cells completed and outputs saved.
     - [`maskgit.py`](/NanoFM_Homeworks/nanofm/models/maskgit.py) with all required sections implemented.
     - `assets/` folder containing the required screenshot images.

3. **nano4M (15%)** — **due 12 April, 23:59**
   - Submit:
     - [`CS503_FM_part3_nano4M.ipynb`](/NanoFM_Homeworks/notebooks/CS503_FM_part3_nano4M.ipynb) with all required cells completed and outputs saved.
     - [`fourm.py`](/NanoFM_Homeworks/nanofm/models/fourm.py) with all required sections implemented.
     - [`assets`](/NanoFM_Homeworks/notebooks/assets) folder containing the required screenshot images.

4. **nanoVLM (7.5%) & nanoFlowMatching (7.5%)** — **due 26 April, 23:59**
   - Submit:
     - Files and detailed instructions will be announced and shared soon.

For **environment setup, training commands, and cluster (SCITAS/IZAR) usage** related to `nanoGPT`, `nanoMaskGIT`, and `nano4M`, please see the detailed instructions in [`NanoFM_Homeworks/README.md`](/NanoFM_Homeworks/README.md).

Please submit all materials via the course Moodle page. Submission links will be released approximately two weeks before each homework deadline.