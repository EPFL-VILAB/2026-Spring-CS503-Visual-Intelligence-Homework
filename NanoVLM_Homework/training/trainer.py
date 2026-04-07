import wandb
from data.processors import get_tokenizer
from training.data import get_dataloaders
from training.scheduler import get_lr
from training.logger import get_run_name
from dataclasses import asdict
from models.vision_language_model import VisionLanguageModel
import torch
import torch.distributed as dist
import torch.optim as optim
import time
import contextlib
from training.evaluation import test_mmstar
from torch.nn.parallel import DistributedDataParallel as DDP
import os


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
        ),
        norm_type,
    )
    return total_norm


def train(train_cfg, vlm_cfg, is_distributed=False, rank=0, world_size=1, local_rank=0):
    train_loader, val_loader, test_loader = get_dataloaders(
        train_cfg, vlm_cfg, is_distributed, rank, world_size
    )
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)

    run = None
    total_dataset_size = len(train_loader.dataset)
    if rank == 0 and train_cfg.log_wandb:
        run_name = get_run_name(train_cfg)
        if train_cfg.data_cutoff_idx is None:
            run_name = run_name.replace("full_ds", f"{total_dataset_size}samples")
        run = wandb.init(
            entity=train_cfg.wandb_entity,
            project="nanoVLM",
            config={"VLMConfig": asdict(vlm_cfg), "TrainConfig": asdict(train_cfg)},
            name=run_name,
        )

    # Initialize model
    start_epoch = 0
    global_step = 0
    best_accuracy = 0
    if train_cfg.resume_from_vlm_checkpoint:
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(
            vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights
        )

    # Define optimizer groups
    # Since we have pretrained vision and language backbones, but a newly initialized modality projection layer, it doesn't make sense to train them with the same learning rate
    # You could opt to fully freeze the backbones and only train the MP layer, but finetuning them with a lower learning rate makes the training as a whole easier
    param_groups = [
        {"params": model.MP.parameters(), "lr": train_cfg.lr_mp},
        {
            "params": list(model.decoder.parameters())
            + list(model.vision_encoder.parameters()),
            "lr": train_cfg.lr_backbones,
        },
    ]
    optimizer = optim.AdamW(param_groups)

    # Calculate steps per epoch correctly
    steps_per_epoch = (
        len(train_loader) + train_cfg.gradient_accumulation_steps - 1
    ) // train_cfg.gradient_accumulation_steps
    max_steps = steps_per_epoch * train_cfg.epochs

    # Load optimizer and training state if resuming
    if train_cfg.resume_from_vlm_checkpoint:
        opt_path = os.path.join(vlm_cfg.vlm_checkpoint_path, "optimizer.pt")
        if os.path.exists(opt_path):
            if rank == 0:
                print(f"Loading optimizer and training state from {opt_path}")
            checkpoint = torch.load(opt_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            global_step = checkpoint.get("global_step", 0)
            best_accuracy = checkpoint.get("best_accuracy", 0)
            start_epoch = global_step // steps_per_epoch
            if rank == 0:
                print(
                    f"Resuming from step {global_step}, estimated epoch {start_epoch}, best accuracy {best_accuracy:.4f}"
                )

    if rank == 0:
        print(
            f"nanoVLM initialized with {sum(p.numel() for p in model.parameters()):,} parameters"
        )
        print(
            f"Training summary: {len(train_loader.dataset)} samples, {len(train_loader)} batches/epoch, global batch size {train_cfg.batch_size * world_size}"
        )
        print(
            f"Steps per epoch: {steps_per_epoch}, Gradient accumulation steps: {train_cfg.gradient_accumulation_steps}"
        )
        print(
            f"Validation summary: {len(val_loader.dataset)} samples, {len(val_loader)} batches/epoch, global batch size {train_cfg.batch_size * world_size}"
        )


    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if train_cfg.compile:
        model = torch.compile(model)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    epoch_times = []
    for epoch in range(start_epoch, train_cfg.epochs):
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        model.train()

        if (epoch+1) == 0:
            print(
                f"[Rank:{rank}] Starting EPOCH {epoch+1}. (Note: If using torch.compile, the first batch will seem 'stuck' for 1-5 minutes while compiling the graph. Let's wait for few minutes!)"
            )
        total_train_loss = 0
        total_tokens_processed = 0

        optimizer.zero_grad()
        # Sum micro-batch stats; log to wandb once per optimizer step (avoids duplicate step= keys when grad acc > 1).
        accum_batch_loss = 0.0
        accum_tokens_per_sec = 0.0
        accum_log_steps = 0
        for i, batch in enumerate(train_loader):
            batch_start_time = time.time()
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if train_cfg.eval_in_epochs and i % 250 == 0:
                model.eval()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    current_model = model
                    if hasattr(current_model, "_orig_mod"):
                        current_model = current_model._orig_mod
                    if is_distributed and hasattr(current_model, "module"):
                        current_model = current_model.module
                    epoch_accuracy = test_mmstar(
                        current_model, tokenizer, test_loader, device
                    )
                    total_val_loss = 0
                    for val_batch in val_loader:
                        val_images = val_batch["image"].to(device)
                        val_input_ids = val_batch["input_ids"].to(device)
                        val_labels = val_batch["labels"].to(device)
                        val_attention_mask = val_batch["attention_mask"].to(device)

                        with torch.amp.autocast(
                            device_type="cuda", dtype=torch.bfloat16
                        ):
                            _, v_loss = model(
                                val_input_ids,
                                val_images,
                                attention_mask=val_attention_mask,
                                targets=val_labels,
                            )
                        total_val_loss += v_loss.item()
                    avg_val_loss = total_val_loss / len(val_loader)

                    if is_distributed:
                        val_loss_tensor = torch.tensor(avg_val_loss, device=device)
                        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                        avg_val_loss = val_loss_tensor.item() / world_size
                model.train()

                if rank == 0:
                    if epoch_accuracy > best_accuracy:
                        best_accuracy = epoch_accuracy
                        current_model.save_pretrained(
                            save_directory=vlm_cfg.vlm_checkpoint_path
                        )
                        opt_save_path = os.path.join(
                            vlm_cfg.vlm_checkpoint_path, "optimizer.pt"
                        )
                        torch.save(
                            {
                                "optimizer_state_dict": optimizer.state_dict(),
                                "global_step": global_step,
                                "best_accuracy": best_accuracy,
                            },
                            opt_save_path,
                        )
                        print(
                            f"Step: {global_step}, Loss: {total_train_loss/max(1, i):.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {epoch_accuracy:.4f} | Saving checkpoint to {vlm_cfg.vlm_checkpoint_path}"
                        )
                    else:
                        print(
                            f"Step: {global_step}, Loss: {total_train_loss/max(1, i):.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
                        )
                    if train_cfg.log_wandb and run is not None:
                        run.log(
                            {
                                "eval/accuracy": epoch_accuracy,
                                "eval/val_loss": avg_val_loss,
                            },
                            step=global_step,
                        )

                if is_distributed:
                    dist.barrier(device_ids=[local_rank])

            # Efficient DDP accumulation: only sync on the last step of the window
            is_accumulating = (i + 1) % train_cfg.gradient_accumulation_steps != 0 and (
                i + 1
            ) != len(train_loader)

            if is_distributed and is_accumulating:
                ddp_model = model
                if hasattr(ddp_model, "_orig_mod"):
                    ddp_model = ddp_model._orig_mod
                sync_context = (
                    ddp_model.no_sync()
                    if hasattr(ddp_model, "no_sync")
                    else contextlib.nullcontext()
                )
            else:
                sync_context = contextlib.nullcontext()

            with sync_context:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(
                        input_ids, images, attention_mask=attention_mask, targets=labels
                    )
                    loss = loss / train_cfg.gradient_accumulation_steps

                loss.backward()

            batch_loss = loss.item() * train_cfg.gradient_accumulation_steps
            total_train_loss += batch_loss

            num_tokens = torch.sum(attention_mask).item()
            num_tokens += (
                images.shape[0]
                * ((images.shape[2] / vlm_cfg.vit_patch_size) ** 2)
                / (vlm_cfg.mp_pixel_shuffle_factor**2)
            )
            total_tokens_processed += num_tokens

            batch_end_time = time.time()
            tokens_per_second = num_tokens / (batch_end_time - batch_start_time)

            accum_batch_loss += batch_loss
            accum_tokens_per_sec += tokens_per_second
            accum_log_steps += 1

            if (i + 1) % train_cfg.gradient_accumulation_steps == 0 or (i + 1) == len(
                train_loader
            ):
                adj_lr_mp = get_lr(global_step, train_cfg.lr_mp, max_steps)
                adj_lr_backbones = get_lr(global_step, train_cfg.lr_backbones, max_steps)
                optimizer.param_groups[0]["lr"] = adj_lr_mp
                optimizer.param_groups[1]["lr"] = adj_lr_backbones

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0 and train_cfg.log_wandb:
                    run.log(
                        {
                            "train/batch_loss": accum_batch_loss / accum_log_steps,
                            "train/tokens_per_second": accum_tokens_per_sec
                            / accum_log_steps,
                            "train/lr_mp": optimizer.param_groups[0]["lr"],
                            "train/lr_backbones": optimizer.param_groups[1]["lr"],
                        },
                        step=global_step,
                    )
                accum_batch_loss = 0.0
                accum_tokens_per_sec = 0.0
                accum_log_steps = 0

        avg_train_loss = total_train_loss / len(train_loader)
        if is_distributed:
            train_loss_tensor = torch.tensor(avg_train_loss, device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = train_loss_tensor.item() / world_size

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        if rank == 0 and train_cfg.log_wandb:
            if run is not None:
                run.log(
                    {
                        "epoch/loss": avg_train_loss,
                        "epoch/duration_s": epoch_duration,
                        "epoch/tokens_per_second": epoch_tokens_per_second,
                    },
                    step=global_step,
                )

            print(
                f"Epoch {epoch + 1}/{train_cfg.epochs}, Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}"
            )

    if rank == 0:
        # Summary Statistics
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_training_time = sum(epoch_times)
        total_samples_processed = len(train_loader.dataset) * train_cfg.epochs
        avg_time_per_sample = total_training_time / total_samples_processed
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"Average time per sample: {avg_time_per_sample:.4f}s")

        current_model = model
        if hasattr(current_model, "_orig_mod"):
            current_model = current_model._orig_mod
        if is_distributed and hasattr(current_model, "module"):
            current_model = current_model.module
        accuracy = test_mmstar(current_model, tokenizer, test_loader, device)
        print(f"MMStar Accuracy: {accuracy:.4f}")

        if train_cfg.log_wandb and run is not None:
            run.summary["avg_epoch_time"] = avg_epoch_time
            run.summary["avg_time_per_sample"] = avg_time_per_sample
            run.summary["mmstar_acc"] = accuracy
            run.finish()
