"""
Contains various utility functions for PyTorch model training and saving.
"""

from argparse import ArgumentParser
import torch
from pathlib import Path


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


def get_args_parser(add_help: bool = True) -> ArgumentParser:
    import argparse

    # Create a parser
    parser = argparse.ArgumentParser(description="Get some hyperparameters.")
    # Get an arg for num_epochs
    parser.add_argument(
        "--num_epochs", default=10, type=int, help="The number of epochs to train for"
    )
    # Get an arg for batch_size
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Number of samples per batch"
    )
    parser.add_argument(
        "--micro_batch_size",
        default=5,
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
        default=600000,
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
        "--delay_lr",
        default=True,
        type=bool,
        help="Whether to delay learning rate scheduling until warmup is complete",
    )
    parser.add_argument(
        "--warmup_iters",
        default=2000,
        type=int,
        help="Number of iterations for learning rate warmup",
    )
    parser.add_argument(
        "--lr_decay_iters",
        default=600000,
        type=int,
        help="Number of iterations over which to decay the learning rate to min_lr",
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
