"""
Handles the model training loop, optimization, and checkpointing.
"""

import os
import torch
from .model import GPT, GPTConfig

class Trainer:
    """
    Manages the training process for the GPT model.
    """
    def __init__(self, model: GPT, optimizer: torch.optim.Optimizer, config: dict, device: str):
        """
        Initializes the Trainer.

        Args:
            model (GPT): The GPT model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            config (dict): The training configuration.
            device (str): The device to run training on ('cpu' or 'cuda').
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device

    def train(self, data_manager):
        """
        Runs the main training loop.

        Args:
            data_manager (DataManager): The data manager providing training batches.
        """
        print("\nStarting training...")
        for step in range(self.config['training']['max_steps']):
            # Get a batch of data
            xb, yb = data_manager.get_batch()

            # Evaluate the loss
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if step % self.config['training']['eval_interval'] == 0:
                print(f"Step {step:4d}/{self.config['training']['max_steps']}: Loss: {loss.item():.4f}")

        print("Training finished.")
        self._save_checkpoint()

    def _save_checkpoint(self):
        """Saves the model state to a checkpoint file."""
        output_dir = self.config['training']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")
