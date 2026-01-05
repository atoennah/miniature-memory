import os
import torch
import torch.nn as nn
from typing import Dict, Any

from .model import GPT, GPTConfig
from .data_loader import DataManager

class Trainer:
    """Manages the model training process."""

    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """
        Initializes the Trainer.

        Args:
            config (Dict[str, Any]): The configuration dictionary.
            data_manager (DataManager): The data manager instance.
        """
        self.config = config
        self.data_manager = data_manager
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.model = self._build_model()
        self.optimizer = self._build_optimizer()

    def _build_model(self) -> nn.Module:
        """
        Builds the GPT model based on the configuration.

        Returns:
            nn.Module: The initialized GPT model.
        """
        gpt_config = GPTConfig(
            vocab_size=self.data_manager.vocab_size,
            block_size=self.config['model']['block_size'],
            n_layer=self.config['model']['n_layer'],
            n_head=self.config['model']['n_head'],
            n_embd=self.config['model']['n_embd'],
            dropout=self.config['model']['dropout']
        )
        model = GPT(gpt_config).to(self.device)
        return model

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Builds the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The initialized optimizer.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate'])
        )
        return optimizer

    def train(self):
        """
        Runs the main training loop.
        """
        print("\nStarting training...")
        for step in range(self.config['training']['max_steps']):
            xb, yb = self.data_manager.get_batch()

            _, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if step % self.config['training']['eval_interval'] == 0:
                print(f"Step {step:4d}/{self.config['training']['max_steps']}: Loss: {loss.item():.4f}")

        print("Training finished.")
        self._save_checkpoint()

    def _save_checkpoint(self):
        """
        Saves the model state dictionary to a checkpoint file.
        """
        output_dir = self.config['training']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")
