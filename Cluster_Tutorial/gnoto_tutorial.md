# Getting started with gnoto


---
# What is gnoto
gnoto is a compute service that provides GPU-powered Jupyter notebooks ([link](https://www.epfl.ch/research/facilities/rcp/gnoto/)). It is well suited for debugging, visualizations, and lightweight compute tasks.

## GPU specifications

Each student can have access to a single Jupyter notebook with 1 V100 GPU with ~10 GB VRAM. The GPU for all gnoto sessions is the same.

# How to create an account
We have already created gnoto accounts for students who are enrolled in the course. Please reach out to the teaching staff in case you face any difficulty in using your account.


# How to access gnoto
The process is straightforward. Login to https://gnoto.epfl.ch/ using your GASPAR credentials, and you will get access to a live Jupyter window.

# Instructions for installing packages

Once logged into the Jupyter notebook, you will find notebook examples for getting started. Follow the instructions on installing new packages using virtual environments.


# FAQ

- **Will my data be retained after I re-login into the gnoto server?**

  Yes. Everything in the `/home/` directory will remain saved.

- **How do I copy/download data into the server?**

  On your own computer, you can use `rsync` [[link](https://scitas-doc.epfl.ch/user-guide/data-management/transferring-data/#using-rsync)] (recommended) or `scp` [[link](https://scitas-doc.epfl.ch/user-guide/data-management/transferring-data/#using-scp)]. Note: these links point to SCITAS documentation, but `rsync` and `scp` are general-purpose data transfer tools that work with any remote server, including gnoto.

