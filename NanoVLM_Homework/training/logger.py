import time

def get_run_name(train_cfg):
    dataset_size = "full_ds" if train_cfg.data_cutoff_idx is None else f"{train_cfg.data_cutoff_idx}samples"
    batch_size = f"bs{train_cfg.batch_size}"
    epochs = f"ep{train_cfg.epochs}"
    learning_rate = f"lr{train_cfg.lr_backbones}-{train_cfg.lr_mp}"
    date = time.strftime("%m%d")

    return f"nanoVLM_{dataset_size}_{batch_size}_{epochs}_{learning_rate}_{date}"
