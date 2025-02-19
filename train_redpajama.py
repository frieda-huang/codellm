# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import sys
import math
import glob
import time
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from codellm.model import LLaMA, LLaMAConfig
from codellm.packed_dataset import PackedDataset, CombinedDataset


out_dir = "out/training"
save_interval = 1000
eval_interval = 1000
eval_iters = 100
log_interval = 1

# compile = False

# Hyperparameters optimized for M2 MacBook with stability fixes
learning_rate = 1e-4  # Further reduced learning rate for stability
batch_size = 32  # Reduced batch size to fit in memory
micro_batch_size = 4  # Smaller micro batches
max_iters = 100000  # Reduced number of iterations for faster training
weight_decay = 1e-2  # Reduced weight decay for stability
beta1 = 0.9
beta2 = 0.999  # Increased beta2 for better stability
grad_clip = 0.5  # Reduced gradient clipping threshold
decay_lr = True
warmup_iters = 2000  # Increased warmup period
lr_decay_iters = max_iters
min_lr = 1e-5


# Data proportions from https://arxiv.org/pdf/2302.13971.pdf Table 1
data_config = [
    ("arxiv", 2.5),
    ("book", 4.5),
    ("c4", 15.0),
    ("cc", 67.0),
    ("github", 4.5),
    ("stackexchange", 2.0),
    ("wikipedia", 4.5),
]


def main(
    train_data_dir: Path = "data/redpajama_train",
    val_data_dir: Path = "data/redpajama_val",
) -> None:
    torch.manual_seed(1337)
    os.makedirs(out_dir, exist_ok=True)

    # Use smaller model configuration for M2 MacBook
    config = LLaMAConfig.from_name("tiny")
    # Reduce context length to save memory
    config.block_size = 512  # Reduced from 2048

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=1338,
    )

    devices = 1
    # Setup device and memory optimizations for M2 MacBook
    device = torch.device("mps")
    # Use mixed precision with float16
    torch.set_default_dtype(torch.float16)
    model = LLaMA(config)
    model = model.float()
    model.apply(model._init_weights)
    model.to(device)
    # Enable memory efficient optimizations
    torch.set_default_dtype(torch.float32)
    # Clear GPU cache
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    # if compile:
    #     model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        foreach=False,
    )

    # Model and optimizer are already set up

    process_batch_size = batch_size // devices
    gradient_accumulation_iters = process_batch_size // micro_batch_size

    train(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        gradient_accumulation_iters,
    )


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    grad_accum_steps: int,
) -> None:
    best_val_loss = float("inf")
    best_model_state = None
    # Initialize weights with smaller values
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param, gain=0.1)

    # Scale learning rate based on gradient accumulation
    optimizer.param_groups[0]["lr"] = learning_rate / grad_accum_steps
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """

    device = next(model.parameters()).device
    step_count = 0
    step_time = 0.0
    tokens = 0
    tokens_sec = 0.0
    prev_t1 = time.time()

    for iter_num, train_data in enumerate(train_dataloader):
        t0 = time.time()

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate

        # update the learning rate in the optimizer
        # the optimizer can have different parameter groups (e.g., different layers might have different learning rates)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        train_data = train_data.to(device)
        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        targets = train_data[:, 1 : model.config.block_size + 1].contiguous()

        is_accumulating = (iter_num + 1) % grad_accum_steps != 0

        logits = model(input_ids)

        # reshape the tensor for `logits` into (batch_size * seq_len, vocab_size), i.e., flattening the batch and sequence dimension into one
        # reshape the targets tensor from (batch_size, seq_len) into a flat array of shape (batch_size * seq_len)
        # `ignore_index=-1` skips computing loss for any position where the target is -1 (often used for padding)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        loss = loss / grad_accum_steps
        if not torch.isfinite(loss):
            print(f"iter {iter_num}: non-finite loss encountered, skipping step update")
            optimizer.zero_grad()
            continue
        loss.backward()

        # Calculate gradient norm for monitoring
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        t1 = time.time()

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            t1 = time.time()

            if val_dataloader is not None and step_count % eval_interval == 0:
                val_loss = validate(model, val_dataloader)
                print(f"step {iter_num}: val loss {val_loss:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "iter_num": iter_num,
                        "step_count": step_count,
                        "val_loss": val_loss,
                    }

            if step_count % save_interval == 0:
                print(f"Saving checkpoint to {out_dir}")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "iter_num": iter_num,
                        "step_count": step_count,
                    },
                    os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"),
                )

        dt = t1 - t0

        tokens += micro_batch_size * model.config.block_size
        step_time += t1 - prev_t1
        prev_t1 = t1

        if iter_num % log_interval == 0:
            tokens_sec_str = f"{tokens / step_time:.0f}" if not is_accumulating else "-"
            print(
                f"iter {iter_num}: loss {loss.item():.4f}, grad_norm {grad_norm:.2f}, lr {lr:.2e}, time: {dt*1000:.2f}ms, speed: {tokens_sec_str} toks/s"
            )

        if not is_accumulating:
            tokens = 0
            step_time = 0.0

        if iter_num > max_iters:
            # Save final model state
            print(f"Saving final model checkpoint to {out_dir}")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "step_count": step_count,
                    "val_loss": best_val_loss if best_model_state else None,
                },
                os.path.join(out_dir, "final_checkpoint.pth"),
            )

            # Save best model if it exists
            if best_model_state is not None:
                print(f"Saving best model (val_loss: {best_val_loss:.4f}) to {out_dir}")
                torch.save(
                    best_model_state, os.path.join(out_dir, "best_checkpoint.pth")
                )
            break


@torch.no_grad()
def validate(model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    print("Validating ...")
    model.eval()
    losses = []
    device = next(model.parameters()).device
    try:
        for k, val_data in enumerate(val_dataloader):
            if k >= eval_iters:
                break
            val_data = val_data.to(device)
            input_ids = val_data[:, 0 : model.config.block_size].contiguous()
            targets = val_data[:, 1 : model.config.block_size + 1].contiguous()
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            losses.append(loss.item())
    except StopIteration:
        print("Warning: Validation dataset exhausted early")

    if not losses:
        print("Warning: No validation loss computed")
        return torch.tensor(float("inf"))

    out = torch.tensor(losses).mean()
    model.train()
    return out


def create_dataloader(
    batch_size: int,
    block_size: int,
    data_dir: str,
    shuffle: bool = True,
    seed: int = 12345,
) -> DataLoader:
    datasets = []
    for prefix, _ in data_config:
        filenames = glob.glob(os.path.join(data_dir, prefix + "*"))
        dataset = PackedDataset(
            filenames,
            n_chunks=4,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed,
            num_processes=1,
            process_rank=0,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )


def create_dataloaders(
    batch_size: int,
    block_size: int,
    train_data_dir: str = "data/redpajama_train",
    val_data_dir: str = "data/redpajama_val",
    seed: int = 12345,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    main()
