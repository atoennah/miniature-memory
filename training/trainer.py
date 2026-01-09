"""
Handles the model training loop, checkpointing, and evaluation.
"""
import os
import torch
from torch.optim import Optimizer

from .model import GPT
from .data_loader import DataManager

class Trainer:
    """Manages the training process for the GPT model."""

    def __init__(self,
                 model: GPT,
                 optimizer: Optimizer,
                 data_manager: DataManager,
                 device: str,
                 max_steps: int,
                 eval_interval: int,
                 output_dir: str):
        """
        Initializes the Trainer.

        Args:
            model (GPT): The GPT model to be trained.
            optimizer (Optimizer): The optimizer for training.
            data_manager (DataManager): The data manager providing training batches.
            device (str): The device to run training on ('cpu' or 'cuda').
            max_steps (int): The total number of training steps to perform.
            eval_interval (int): The interval (in steps) at which to print loss.
            output_dir (str): The directory to save model checkpoints.
        """
        self.model = model
        self.optimizer = optimizer
        self.data_manager = data_manager
        self.device = device
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self):
        """Runs the main training loop."""
        print("\nStarting training...")
        for step in range(self.max_steps):
            # Get a batch of data
            xb, yb = self.data_manager.get_batch()

            # Evaluate the loss
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if step % self.eval_interval == 0:
                print(f"Step {step:4d}/{self.max_steps}: Loss: {loss.item():.4f}")

        print("Training finished.")
        self._save_checkpoint()

    def _save_checkpoint(self):
        """Saves the model state to a checkpoint file."""
        checkpoint_path = os.path.join(self.output_dir, 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")
