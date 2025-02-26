"""
Contains functions for training and testing a PyTorch model.
"""

import math
import time
from dataclasses import dataclass
from typing import Tuple

import torch
import utils
from rich import print
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Optimization
    learning_rate: float
    grad_clip: float
    decay_lr: bool

    # Learning rate scheduler
    warmup_iters: int
    lr_decay_iters: int
    min_lr: float

    # Batch sizes and steps
    batch_size: int
    grad_accum_steps: int
    max_iters: int
    eval_iters: int

    # Intervals
    save_interval: int
    eval_interval: int
    log_interval: int


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
    writer: torch.utils.tensorboard.writer.SummaryWriter,
) -> torch.utils.tensorboard.writer.SummaryWriter:
    model.train()  # Put model in train mode

    step_count = 0  # Track completed optimization steps

    step_time = 0.0  # Track time spent during current gradient accumulation window
    tokens = 0  # Track tokens processed during current gradient accumulation window
    prev_t1 = time.time()  # Initial timestamp

    print(f"Training for maximum {config.max_iters} iterations")

    # Loop through data loader data batches
    for iter_num, train_data in enumerate(tqdm(train_dataloader)):
        t0 = time.time()

        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num, config) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Send data to target device
        train_data = train_data.to(device)

        seq_len = min(model.config.block_size, train_data.size(1) - 1)
        input_ids = train_data[:, 0:seq_len].contiguous()
        targets = train_data[:, 1 : seq_len + 1].contiguous()

        is_accumulating = (iter_num + 1) % config.grad_accum_steps != 0

        # 1. Forward pass
        logits = model(input_ids)

        # 2. Calculate and accumulate loss
        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )

        # 3. Loss backward
        (loss / config.grad_accum_steps).backward()

        t1 = time.time()

        # Apply gradient clipping and update model parameters if gradient accumulation is complete
        if not is_accumulating:
            # Clip gradients
            clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)

            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            t1 = time.time()

            # Run validation
            if val_dataloader is not None and step_count % config.eval_interval == 0:
                val_loss = validate(
                    model,
                    val_dataloader,
                    loss_fn,
                    device,
                    config.eval_iters,
                )
                print(f"step {iter_num}: val loss {val_loss:.4f}")

                ### New: Experiment tracking ###
                # Add loss results to SummaryWriter
                writer.add_scalars(
                    main_tag="Loss",
                    tag_scalar_dict={"train_loss": loss, "val_loss": val_loss},
                    global_step=iter_num,
                )

            # Save model checkpoint
            if step_count % config.save_interval == 0:
                utils.save_model(
                    model=model,
                    target_dir="checkpoints/model",
                    model_name=f"iter-{iter_num:06d}-ckpt.pt",
                )

        dt = t1 - t0

        # Default: 5 * 512 = 2560 tokens
        tokens += config.batch_size * model.config.block_size
        step_time += t1 - prev_t1  # Accumulate total time across iterations
        prev_t1 = t1  # Update for next iteration

        # Logging
        if iter_num % config.log_interval == 0:
            tokens_sec_str = f"{tokens / step_time:.0f}" if not is_accumulating else "-"
            print(
                f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms, speed: {tokens_sec_str} toks/s/device"
            )

        # Reset tracking
        if not is_accumulating:
            tokens = 0
            step_time = 0.0

        if iter_num > config.max_iters:
            break

    writer.close()


def validate(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    eval_iters: int,
) -> Tuple[float, float]:
    print("Validating...")
    # Put model in eval mode
    model.eval()
    losses = torch.zeros(eval_iters)
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for k, val_data in enumerate(val_dataloader):
            if k >= eval_iters:  # Only evaluate up to eval_iters batches
                break

            # Send data to target device
            val_data = val_data.to(device)

            # Get sequences of block_size length
            seq_len = min(model.config.block_size, val_data.size(1) - 1)
            input_ids = val_data[:, :seq_len].contiguous()
            targets = val_data[:, 1 : seq_len + 1].contiguous()

            # 1. Forward pass
            logits = model(input_ids)

            # 2. Calculate loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

            losses[k] = loss.item()

    out = losses.mean()
    model.train()

    return out


def get_lr(it: int, config: TrainingConfig):
    """Learning rate decay scheduler (cosine with warmup)"""
    warmup_iters = config.warmup_iters
    learning_rate = config.learning_rate
    lr_decay_iters = config.lr_decay_iters
    min_lr = config.min_lr

    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    # 2) If it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr

    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)

    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Coeff ranges 0..1

    return min_lr + coeff * (learning_rate - min_lr)
