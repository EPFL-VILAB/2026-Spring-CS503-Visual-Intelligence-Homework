import time
import torch
import random
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

torch.manual_seed(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
import models.config as config
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from training.trainer import train


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Fallback for single GPU/CPU
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return True, rank, world_size, local_rank

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    is_distributed, rank, world_size, local_rank = setup_distributed()

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_mp', type=float, help='Learning rate for the mapping network')
    parser.add_argument('--lr_backbones', type=float, help='Learning rate for the backbones')
    parser.add_argument('--vlm_checkpoint_path', type=str, help='Path to the VLM checkpoint for loading or saving')
    parser.add_argument('--resume_from_vlm_checkpoint', type=bool, default=False, help='Resume training from VLM checkpoint specified by vlm_checkpoint_path (or default if not provided)')

    args = parser.parse_args()

    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig()

    if args.lr_mp is not None:
        train_cfg.lr_mp = args.lr_mp
    if args.lr_backbones is not None:
        train_cfg.lr_backbones = args.lr_backbones
    if args.vlm_checkpoint_path is not None:
        vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path

    if args.resume_from_vlm_checkpoint and args.vlm_checkpoint_path is not None:
        train_cfg.resume_from_vlm_checkpoint = True
        # When resuming a full VLM, we don't need to load individual backbone weights from original sources
        vlm_cfg.vlm_load_backbone_weights = False

    if rank == 0:
        print("--- VLM Config ---")
        print(vlm_cfg)
        print("--- Train Config ---")
        print(train_cfg)

    try:
        train(train_cfg, vlm_cfg, is_distributed, rank, world_size, local_rank)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()