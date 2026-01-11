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
Handles the training loop for the GPT model.
"""
import os
import torch

class Trainer:
    """Manages the model, optimizer, and the training loop."""

    def __init__(self, config, model, optimizer, data_manager):
import os
import torch
from .model import GPT, GPTConfig

class Trainer:
    """Manages the model training loop."""
    def __init__(self, config, data_manager, device):
        self.config = config
        self.data_manager = data_manager
        self.device = device
        print(f"Using device: {self.device}")

        # Create output directory
        os.makedirs(self.config['training']['output_dir'], exist_ok=True)

        # Build the model
    """
    Manages the model, optimizer, and the training loop.
    This class encapsulates the core training mechanics, allowing the main
    script to be a simple orchestrator.
    """
    def __init__(self, config, vocab_size, device):
        """
        Initializes the Trainer.
        Args:
            config (dict): A dictionary containing the model and training configuration.
            vocab_size (int): The size of the vocabulary.
            device (str): The device to train on ('cpu' or 'cuda').
        """
        self.config = config
        self.device = device
        self.model = self._create_model(vocab_size)
        self.optimizer = self._create_optimizer()

    def _create_model(self, vocab_size):
        """Creates the GPT model based on the configuration."""
        model_config = self.config['model']
        gpt_config = GPTConfig(
            vocab_size=vocab_size,
            block_size=model_config['block_size'],
            n_layer=model_config['n_layer'],
            n_head=model_config['n_head'],
            n_embd=model_config['n_embd'],
            dropout=model_config['dropout']
        )
        return GPT(gpt_config).to(self.device)

    def _create_optimizer(self):
        """Creates the AdamW optimizer for the model."""
from .data_loader import DataManager
from typing import Dict

class Trainer:
    """Manages the model training loop."""

    def __init__(self, config: Dict, data_manager: DataManager):
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
            config: The training configuration.
            model: The GPT model instance.
            optimizer: The optimizer instance.
            data_manager: The DataManager instance.
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.data_manager = data_manager

    def train(self):
        """Runs the main training loop."""
        print("\nStarting training...")
        for step in range(self.max_steps):
        for step in range(self.config['training']['max_steps']):
            # Get a batch of data
            xb, yb = self.data_manager.get_batch()

            # Evaluate the loss
            config (Dict): The configuration dictionary.
import torch.nn as nn
from typing import Dict, Any

from .model import GPT, GPTConfig
from .data_loader import DataManager
"""
class Trainer:
    """Manages the model training process."""

    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """
        Initializes the Trainer.
        Orchestrates the model training process.

    This class encapsulates the training loop, model and optimizer setup,
    and checkpointing. It is designed to be configured via a dictionary.

    Attributes:
        config (Dict[str, Any]): The configuration dictionary.
        device (str): The computing device ('cuda' or 'cpu').
        model (GPT): The GPT model instance.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
    """

    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """Initializes the Trainer.

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
        self._setup_model()
        self._setup_optimizer()

    def _setup_model(self) -> None:
        """Initializes the GPT model based on the configuration."""
        gpt_config = GPTConfig(
            vocab_size=self.data_manager.vocab_size,
            block_size=self.config['model']['block_size'],
            n_layer=self.config['model']['n_layer'],
            n_head=self.config['model']['n_head'],
            n_embd=self.config['model']['n_embd'],
            dropout=self.config['model']['dropout']
        )
        self.model = GPT(gpt_config).to(self.device)

        # Create the optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate'])
        )

    def train(self):
        """Runs the training loop for the specified number of steps."""
        print("\nStarting training...")
        for step in range(self.config['training']['max_steps']):
            # Get a batch of data
            xb, yb = self.data_manager.get_batch()

            # Evaluate the loss
        self.optimizer = self._configure_optimizer()

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """Configures the AdamW optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate'])
        )

    def train(self, data_manager):
        """
        Runs the main training loop.
        Args:
            data_manager (DataManager): The data manager instance for fetching batches.
        """
        print("\nStarting training...")
        for step in range(self.config['training']['max_steps']):
            xb, yb = data_manager.get_batch()

            # Evaluate the loss
    def train(self):
        """Runs the main training loop."""
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
        self.model = GPT(gpt_config).to(self.device)
        print("Model configured and moved to device.")

    def _setup_optimizer(self) -> None:
        """Initializes the AdamW optimizer."""
        learning_rate = float(self.config['training']['learning_rate'])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        print(f"Optimizer configured with learning rate: {learning_rate}")

    def run(self) -> None:
        """Executes the main training loop."""
        print("\nStarting training...")
        max_steps = self.config['training']['max_steps']
        eval_interval = self.config['training']['eval_interval']

        for step in range(max_steps):
            xb, yb = self.data_manager.get_batch()

            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if step % self.eval_interval == 0:
                print(f"Step {step:4d}/{self.max_steps}: Loss: {loss.item():.4f}")
            if step % self.config['training']['eval_interval'] == 0:
                print(f"Step {step:4d}/{self.config['training']['max_steps']}: Loss: {loss.item():.4f}")

        print("Training finished.")
        self._save_checkpoint()

    def _save_checkpoint(self):
        """Saves the model checkpoint."""
        checkpoint_path = os.path.join(self.config['training']['output_dir'], 'model.pt')
        """Saves the model state to a checkpoint file."""
        checkpoint_path = os.path.join(self.output_dir, 'model.pt')
        """Saves the model checkpoint."""
        """
        Saves the model state dictionary to a checkpoint file.
        """
            if step % eval_interval == 0:
                print(f"Step {step:4d}/{max_steps}: Loss: {loss.item():.4f}")

        print("Training finished.")
        self.save_checkpoint()

    def save_checkpoint(self) -> None:
        """Saves the model state to a checkpoint file."""
        output_dir = self.config['training']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")
