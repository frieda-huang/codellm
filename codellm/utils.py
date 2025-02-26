"""
Contains various utility functions for PyTorch model training and saving.
"""

from argparse import ArgumentParser
from pathlib import Path

import torch
from rich import print
from torch.utils.tensorboard import SummaryWriter


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
      model: A target PyTorch model to save.
      target_dir: A directory for saving the model to.
      model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
      save_model(model=model_0,
                 target_dir="models",
                 model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def set_seeds(seed: int = 42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if torch.mps.is_available():
        torch.mps.manual_seed(seed)


def create_writer(experiment_name: str, model_name: str, extra: str = None):
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    import os
    from datetime import datetime

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime(
        "%Y-%m-%d"
    )  # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


def get_args_parser(add_help: bool = True) -> ArgumentParser:
    import argparse

    # Create a parser
    parser = argparse.ArgumentParser(description="Get some hyperparameters.")
    # Get an arg for batch_size
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Number of samples per batch"
    )
    parser.add_argument(
        "--micro_batch_size",
        default=4,
        type=int,
        help="Number of samples per micro batch",
    )
    parser.add_argument(
        "--block_size",
        default=512,
        type=int,
        help="Maximum sequence length for the model's context window",
    )
    # Get an arg for learning_rate
    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        help="Learning rate to use for model",
    )
    parser.add_argument(
        "--train_dir",
        default="data/redpajama_train",
        type=str,
        help="Directory containing training data files",
    )

    parser.add_argument(
        "--val_dir",
        default="data/redpajama_val",
        type=str,
        help="Directory containing validation data files",
    )

    parser.add_argument(
        "--max_iters",
        default=100,
        type=int,
        help="Maximum number of training iterations",
    )
    parser.add_argument(
        "--beta1",
        default=0.9,
        type=float,
        help="AdamW optimizer beta1 (decay rate for 1st moment)",
    )
    parser.add_argument(
        "--beta2",
        default=0.95,
        type=float,
        help="AdamW optimizer beta2 (decay rate for 2nd moment)",
    )
    parser.add_argument(
        "--grad_clip",
        default=1.0,
        type=float,
        help="Gradient clipping threshold to prevent exploding gradients",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.1,
        type=float,
        help="Weight decay coefficient for L2 regularization (default: 0.1). Controls the strength of parameter regularization to prevent overfitting",
    )

    parser.add_argument(
        "--decay_lr",
        default=True,
        type=bool,
        help="Whether to use learning rate scheduling (linear warmup followed by cosine decay) or keep a constant learning rate",
    )
    parser.add_argument(
        "--warmup_iters",
        default=2000,
        type=int,
        help="Number of iterations for learning rate warmup",
    )
    parser.add_argument(
        "--lr_decay_iters",
        default=80,  # Set to 80% of max_iters (1000)
        type=int,
        help="Number of iterations over which to decay the learning rate to min_lr",
    )
    # Training interval parameters
    parser.add_argument(
        "--save_interval",
        default=5,
        type=int,
        help="Save model checkpoint every N iterations",
    )
    parser.add_argument(
        "--eval_interval",
        default=5,
        type=int,
        help="Evaluate model on validation set every N iterations",
    )
    parser.add_argument(
        "--eval_iters",
        default=100,
        type=int,
        help="Number of validation batches to evaluate on during evaluation",
    )
    parser.add_argument(
        "--log_interval",
        default=1,
        type=int,
        help="Print training metrics every N iterations",
    )
    parser.add_argument(
        "--min_lr",
        default=6e-5,
        type=float,
        help="Minimum learning rate after decay",
    )
    parser.add_argument(
        "--num_devices",
        default=1,
        type=int,
        help="Number of devices for distributed training. Used to calculate per-device batch size and gradient accumulation steps",
    )

    return parser
