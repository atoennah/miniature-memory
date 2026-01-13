"""
This module contains the Trainer class, which encapsulates the core logic for
training the GPT model. It handles model and optimizer initialization, the main
training loop, validation, and checkpointing.
"""
import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from typing import Dict, Any

from .data_loader import DataManager
from .model import GPT, GPTConfig

class Trainer:
    """
    Orchestrates the model training, validation, and checkpointing process.

    This class encapsulates the entire training pipeline, from building the model
    and optimizer to executing the training loop and saving checkpoints. It is
    designed to be configured via a YAML file.

    Attributes:
        config (Dict[str, Any]): The configuration dictionary.
        data_manager (DataManager): An instance of the data manager.
        device (str): The computing device ('cuda' or 'cpu').
        model (nn.Module): The GPT model instance.
        optimizer (torch.optim.Optimizer): The AdamW optimizer.
        scaler (GradScaler): The gradient scaler for mixed-precision training.
    """

    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """
        Initializes the Trainer.

        Args:
            config (Dict[str, Any]): The configuration dictionary.
            data_manager (DataManager): An instance of the data manager.
        """
        self.config: Dict[str, Any] = config
        self.data_manager: DataManager = data_manager
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.model: nn.Module = self._build_model()
        self.optimizer: torch.optim.Optimizer = self._build_optimizer()
        self.scaler: GradScaler = GradScaler(enabled=(self.device == 'cuda'))

    def _build_model(self) -> nn.Module:
        """
        Builds the GPT model based on the configuration.

        Returns:
            An instance of the GPT model, moved to the configured device.
        """
        model_config: Dict[str, Any] = self.config['model']
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
        """
        Builds the AdamW optimizer for the model.

        Returns:
            An instance of the AdamW optimizer.
        """
        learning_rate: float = float(self.config['training']['learning_rate'])
        return torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def _train_step(self) -> float:
        """
        Performs a single training step, including forward and backward passes.

        This method fetches a batch of training data, performs a forward pass,
        computes the loss, and updates the model weights using mixed-precision
        training and gradient clipping.

        Returns:
            The training loss for the current step.
        """
        xb, yb = self.data_manager.get_batch('train')

        with torch.amp.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.device == 'cuda')):
            _, loss = self.model(xb, yb)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()

        grad_clip: float = self.config['training'].get('grad_clip', 1.0)
        if grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    @torch.no_grad()
    def _validate(self) -> float:
        """
        Calculates the validation loss over a configured number of steps.

        This method sets the model to evaluation mode and computes the average
        loss over a set of validation batches.

        Returns:
            The average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        val_steps: int = self.config['training'].get('val_steps', 20)
        for _ in range(val_steps):
            xb, yb = self.data_manager.get_batch('val')
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.device == 'cuda')):
                _, loss = self.model(xb, yb)
            total_loss += loss.item()
        self.model.train()
        return total_loss / val_steps

    def run(self) -> None:
        """
        Executes the main training loop.

        This method orchestrates the training process, including running training
        steps, periodically validating the model, and logging progress. It
        concludes by saving a final model checkpoint.
        """
        print("\nStarting training...")
        start_time: float = time.time()
        max_steps: int = self.config['training']['max_steps']
        eval_interval: int = self.config['training']['eval_interval']

        for step in range(max_steps):
            train_loss = self._train_step()
            if step % eval_interval == 0 or step == max_steps - 1:
                val_loss = self._validate()
                print(f"Step {step:4d}/{max_steps}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        end_time: float = time.time()
        duration: float = end_time - start_time
        print(f"Training finished in {duration:.2f} seconds.")
        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        """
        Saves the model's state dictionary to a checkpoint file.

        The checkpoint is saved in the directory specified by the `output_dir`
        configuration parameter.
        """
        output_dir: str = self.config['training']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path: str = os.path.join(output_dir, 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")
