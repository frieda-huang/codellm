from argparse import ArgumentParser

import data_setup
import engine
import model_builder
import torch
import utils
from rich import print


def main(args: ArgumentParser):
    # Setup hyperparameters
    NUM_DEVICES = args.num_devices
    BATCH_SIZE = args.batch_size
    MICRO_BATCH_SIZE = args.micro_batch_size
    LEARNING_RATE = args.learning_rate
    BLOCK_SIZE = args.block_size
    WEIGHT_DECAY = args.weight_decay
    MAX_ITERS = args.max_iters
    BETA1 = args.beta1
    BETA2 = args.beta2
    GRAD_CLIP = args.grad_clip
    DECAY_LR = args.decay_lr
    WARMUP_ITERS = args.warmup_iters
    LR_DECAY_ITERS = args.lr_decay_iters
    MIN_LR = args.min_lr

    SAVE_INTERVAL = args.save_interval
    EVAL_INTERVAL = args.eval_interval
    LOG_INTERVAL = args.log_interval
    EVAL_ITERS = args.eval_iters

    print(
        f"[INFO] Training a model for {MAX_ITERS} max iterations with batch size {BATCH_SIZE} and a learning rate of {LEARNING_RATE}"
    )

    # Setup directories
    train_dir = args.train_dir
    val_dir = args.val_dir
    print(f"[INFO] Training data file: {train_dir}")
    print(f"[INFO] Validation data file: {val_dir}")

    # Setup target device - prefer CUDA over MPS over CPU
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"[INFO] Using device: {device}")

    train_dataloader, val_dataloader = data_setup.create_dataloaders(
        train_data_dir=train_dir,
        val_data_dir=val_dir,
        batch_size=MICRO_BATCH_SIZE,
        block_size=BLOCK_SIZE,
    )

    # Create model with tiny config to train on a M2 MacBook
    config = model_builder.LLaMAConfig.from_name("tiny")
    model = model_builder.LLaMA(config)
    model.apply(model._init_weights)
    torch.set_default_dtype(torch.float32)

    # Move model to device
    model = model.to(device)

    # Set loss and optimizer
    # Note we don't need to pass `ignore_index` every time
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(BETA1, BETA2),
        foreach=False,
    )

    # Print out training progress
    print(f"[INFO] Training model for {MAX_ITERS} max iterations...")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Batch size: {BATCH_SIZE}")
    print(f"[INFO] Learning rate: {LEARNING_RATE}")

    process_batch_size = BATCH_SIZE // NUM_DEVICES
    grad_accum_steps = process_batch_size // MICRO_BATCH_SIZE

    training_config = engine.TrainingConfig(
        # Optimization
        learning_rate=LEARNING_RATE,
        grad_clip=GRAD_CLIP,
        decay_lr=DECAY_LR,
        # Learning rate scheduler
        warmup_iters=WARMUP_ITERS,
        lr_decay_iters=LR_DECAY_ITERS,
        min_lr=MIN_LR,
        # Batch sizes and steps
        batch_size=MICRO_BATCH_SIZE,
        grad_accum_steps=grad_accum_steps,
        max_iters=MAX_ITERS,
        # Intervals
        save_interval=SAVE_INTERVAL,
        eval_interval=EVAL_INTERVAL,
        eval_iters=EVAL_ITERS,
        log_interval=LOG_INTERVAL,
    )

    writer = utils.create_writer(
        experiment_name="verify_inputs",
        model_name="llama2-tiny",
        extra="batch_size256",
    )

    utils.set_seeds()
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        config=training_config,
        writer=writer,
    )


if __name__ == "__main__":
    args = utils.get_args_parser().parse_args()
    main(args)
