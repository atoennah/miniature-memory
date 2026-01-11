"""
This module contains the Trainer class, which encapsulates the core logic
for training the GPT model. It handles model and optimizer initialization,
the main training loop, and saving checkpoints.
"""
import os
import time
import torch
import torch.nn as nn
from typing import Dict, Any

from .data_loader import DataManager
from .model import GPT, GPTConfig

class Trainer:
    """
    Orchestrates the model training process.

    This class encapsulates the training loop, model and optimizer setup,
    and checkpointing. It is designed to be configured via a dictionary.

    Attributes:
        config (Dict[str, Any]): The configuration dictionary.
        data_manager (DataManager): The data manager instance.
        device (str): The computing device ('cuda' or 'cpu').
        model (GPT): The GPT model instance.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
    """

    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """Initializes the Trainer.

        Args:
            config: The configuration dictionary.
            data_manager: The data manager instance.
        """
        self.config = config
        self.data_manager = data_manager
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Initialize model and optimizer
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()

    def _build_model(self) -> nn.Module:
        """Builds the GPT model based on the configuration."""
        model_config = self.config['model']
        gpt_config = GPTConfig(
            vocab_size=self.data_manager.vocab_size,
            block_size=model_config['block_size'],
            n_layer=model_config['n_layer'],
            n_head=model_config['n_head'],
            n_embd=model_config['n_embd'],
            dropout=model_config['dropout']
        )
        return GPT(gpt_config).to(self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Builds the AdamW optimizer for the model."""
        learning_rate = float(self.config['training']['learning_rate'])
        return torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def run(self) -> None:
        """Executes the main training loop with mixed precision and gradient clipping."""
        print("\nStarting training...")
        start_time = time.time()
        max_steps = self.config['training']['max_steps']
        eval_interval = self.config['training']['eval_interval']
        grad_clip = self.config['training'].get('grad_clip', 1.0)

        # Use AMP scaler if on CUDA for performance
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

        for step in range(max_steps):
            xb, yb = self.data_manager.get_batch()

            # Forward pass with Automatic Mixed Precision
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.device == 'cuda')):
                logits, loss = self.model(xb, yb)

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Gradient Clipping to prevent explosions
            if grad_clip > 0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            # Update weights
            scaler.step(self.optimizer)
            scaler.update()

            # Log progress
            if step % eval_interval == 0 or step == max_steps - 1:
                print(f"Step {step:4d}/{max_steps}: Loss: {loss.item():.4f}")

        end_time = time.time()
        duration = end_time - start_time
        print(f"Training finished in {duration:.2f} seconds.")
        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        """Saves the model's state dictionary to a checkpoint file."""
        output_dir = self.config['training']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")
