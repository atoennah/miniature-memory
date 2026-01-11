"""
This script serves as the main entry point for the training pipeline.

It orchestrates the data loading, model initialization, and the training
process by leveraging the DataManager and Trainer classes.
"""

Main entry point for training the GPT model.
"""
Main script to run the GPT model training process.

This script acts as the orchestrator, bringing together the data manager,
model, and trainer to execute the training loop based on a provided
configuration file.
"""
import torch
from .data_loader import DataManager
from .trainer import Trainer

def main(config):
    """
    Main function to orchestrate the training process.
    This function initializes the data manager and the trainer, then
    starts the training loop.
    Args:
        config (dict): A dictionary containing the full training configuration.
        """
import os
import argparse
import os
import time
import torch
import yaml
from typing import Dict, Any

from .model import GPT, GPTConfig
from .data_loader import DataManager
import sys
import yaml
from typing import Tuple, List, Dict, Any, Callable

import torch
import time
import torch
import yaml
import argparse
from typing import Dict, Any

from .model import GPT, GPTConfig
from .data_loader import DataManager

class Trainer:
    """Manages the model, optimizer, and the core training loop."""

    def __init__(self, config: Dict[str, Any], model: GPT, optimizer: torch.optim.Optimizer, data_manager: DataManager):
        """
        Initializes the Trainer.

        Args:
            config (Dict[str, Any]): The training configuration.
            model (GPT): The GPT model to train.
            optimizer (torch.optim.Optimizer): The optimizer.
            data_manager (DataManager): The data manager.
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.data_manager = data_manager

    def train(self):
        """Runs the main training loop."""
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
        """Saves the model checkpoint."""
        output_dir = self.config['training']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")

def run_training(config: Dict[str, Any]):
    """
    Sets up and runs the training process.

    Args:
        config (Dict[str, Any]): The training configuration.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Data ---
import argparse
import yaml
import torch
from .model import GPT, GPTConfig
from .data_loader import DataManager
from .trainer import Trainer

def run_training(config: dict):
    """
    Orchestrates the model training process.

    Args:
        config (dict): A dictionary containing the training configuration.

from .data_loader import DataManager
from .trainer import Trainer

def run_training(config):
    """
    Orchestrates the training process by initializing the data manager and trainer.
    """
    # Determine the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the data manager

from .model import GPT, GPTConfig
from .data_loader import DataManager
from .trainer import Trainer

def main(config: dict):
    """
    Orchestrates the training process from configuration.

    Args:
        config (dict): A dictionary containing the training configuration.
import torch

from .data_loader import DataManager
from .model import GPT, GPTConfig
from .trainer import Trainer

def run_training(config: dict):
    """
    Orchestrates the training process from configuration.

    Args:
        config (dict): A dictionary containing the training configuration.

from .data_loader import DataManager
from .trainer import Trainer
from typing import Dict

def run_training(config: Dict):
    """
    Initializes and runs the training process.

    Args:
        config (Dict): The configuration dictionary for training.
from typing import Dict, Any
import torch

from .data_loader import DataManager
from .trainer import Trainer

def main(config: Dict[str, Any]):
    """
    Main function to run the training pipeline.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
    """
    # --- Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Data ---
from typing import Dict, Any, Tuple, List, Callable

from .model import GPT, GPTConfig
from typing import Dict, Any, Tuple

def get_data(data_path: str) -> Tuple[torch.Tensor, int]:
    """Reads training data and returns it as a tensor."""
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    data = torch.tensor(encode(text), dtype=torch.long)
    return data, vocab_size

def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a small batch of data of inputs x and targets y."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Stack the input sequences into a batch
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Stack the target sequences (shifted by one) into a batch
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --- Main Training Loop ---

def run_training(config: Dict[str, Any]) -> None:
    """
    Executes the main training loop for the GPT model.

    This function orchestrates the entire training process, including:
    - Setting up the device (CPU/GPU).
    - Loading and preparing the data.
    - Initializing the model and optimizer.
    - Running the training steps.
    - Saving the final model checkpoint.

    Args:
        config (Dict[str, Any]): A dictionary containing the full training configuration,
                                 typically loaded from a YAML file.
    """
    # --- Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Data Manager ---
    data_manager = DataManager(
        data_path=config['data']['path'],
        block_size=config['model']['block_size'],
        batch_size=config['training']['batch_size'],
        device=device
    )
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

# --- Trainer Class ---

class Trainer:
    """
    The Trainer class encapsulates the logic for training a GPT model.

    This class handles the initialization of the model, data loading, the main
    training loop, and checkpointing. By encapsulating the training process,
    it provides a clean and reusable interface for running training sessions.
    """
    def __init__(self, config: Dict[str, Any]):
        """Initializes the Trainer."""
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        os.makedirs(self.config['training']['output_dir'], exist_ok=True)

        self.data, self.vocab_size = get_data(self.config['data']['path'])

        gpt_config = GPTConfig(
            vocab_size=self.vocab_size,
            block_size=self.config['model']['block_size'],
            n_layer=self.config['model']['n_layer'],
            n_head=self.config['model']['n_head'],
            n_embd=self.config['model']['n_embd'],
            dropout=self.config['model']['dropout']
        )
        return GPT(gpt_config).to(self.device)

    def train(self) -> None:
        """
        Runs the main training loop for the specified number of steps.

        This method iterates through the training data, performs forward and
        backward passes, and updates the model parameters. It also logs the
        training loss at specified intervals.
        """
        print("\nStarting training...")
        for step in range(self.config['training']['max_steps']):
            # Get a batch of data
            xb, yb = get_batch(
                self.data,
                self.model.config.block_size,
                self.config['training']['batch_size'],
                self.device
            )

            # Perform a forward and backward pass
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # Log the training loss
            if step % self.config['training']['eval_interval'] == 0:
                print(f"Step {step:4d}/{self.config['training']['max_steps']}: Loss: {loss.item():.4f}")

        print("Training finished.")

    # --- Model ---
    model_config = config['model']
    gpt_config = GPTConfig(
        vocab_size=data_manager.vocab_size,
        vocab_size=vocab_size,
        block_size=model_config['block_size'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        n_embd=model_config['n_embd'],
        dropout=model_config['dropout']
    def save_checkpoint(self) -> None:
        """
        Saves the model's state dictionary to a checkpoint file.

        The checkpoint is saved in the output directory specified in the
        training configuration.
        """
        checkpoint_path = os.path.join(self.config['training']['output_dir'], 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")


# --- Main Execution Block ---
class Trainer:
    """
    A class to encapsulate the training loop for the GPT model.
    It handles model initialization, optimization, and the training process.
    """
    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """
        Initializes the Trainer.

        Args:
            config: A dictionary containing the configuration for the trainer.
            data_manager: An instance of DataManager that handles data loading.
        """
        self.config = config
        self.data_manager = data_manager
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Create output directory if it doesn't exist
        os.makedirs(self.config['training']['output_dir'], exist_ok=True)

        # Get vocab_size from the data_manager
        vocab_size = self.data_manager.vocab_size

        # Initialize model
        gpt_config = GPTConfig(
            vocab_size=vocab_size,
            block_size=self.config['model']['block_size'],
            n_layer=self.config['model']['n_layer'],
            n_head=self.config['model']['n_head'],
            n_embd=self.config['model']['n_embd'],
            dropout=self.config['model']['dropout']
        )
        self.model = GPT(gpt_config).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate'])
        )

    def _run_step(self, xb: torch.Tensor, yb: torch.Tensor, step: int):
        """Performs a single training step."""
        logits, loss = self.model(xb, yb)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if step % self.config['training']['eval_interval'] == 0:
            print(f"Step {step:4d}/{self.config['training']['max_steps']}: Loss: {loss.item():.4f}")

    def _save_checkpoint(self):
        """Saves the model checkpoint."""
        checkpoint_path = os.path.join(self.config['training']['output_dir'], 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")

    def run(self):
        """Runs the main training loop."""
        print("\nStarting training...")
        start_time = time.time()
        for step in range(self.config['training']['max_steps']):
            xb, yb = get_batch(
                self.data,
                self.model.config.block_size,
                self.config['training']['batch_size'],
                self.device
            )
            self._run_step(xb, yb, step)

        end_time = time.time()
        duration = end_time - start_time
        print(f"Training finished in {duration:.2f} seconds.")
        self._save_checkpoint()

    def _run_step(self, xb: torch.Tensor, yb: torch.Tensor, step: int) -> None:
        """Performs a single training step."""
        logits, loss = self.model(xb, yb)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if step % training_config['eval_interval'] == 0:
            print(f"Step {step:4d}/{training_config['max_steps']}: Loss: {loss.item():.4f}")
        if step % self.config['training']['eval_interval'] == 0:
            print(f"Step {step:4d}/{self.config['training']['max_steps']}: Loss: {loss.item():.4f}")

    def _save_checkpoint(self) -> None:
        """Saves the model checkpoint."""
        checkpoint_path = os.path.join(self.config['training']['output_dir'], 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")

    # --- Save Checkpoint ---
    checkpoint_path = os.path.join(output_dir, 'model.pt')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModel checkpoint saved to: {checkpoint_path}")

def main() -> None:
    """
    Provides a command-line interface for running the training script.

    This function handles argument parsing, configuration loading, and applying
    hardcoded defaults for a quick "micro-train" test run.
    """
def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main() -> None:
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description="Train a NanoGPT model.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/small.yaml',
        help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()

    # Construct the full path to the config file
    # This assumes the script is run from the root of the repository
    config_path = os.path.join('training', args.config)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Hardcoded values for standalone execution
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
        exit(1)
        print(f"Error: Config file not found at {args.config}", file=sys.stderr)
        sys.exit(1)

    # For standalone execution, provide sensible defaults for a quick micro-train.
    # These values are suitable for a quick test run.
    training_defaults = {
        'max_steps': 100,
        'eval_interval': 10,
        'output_dir': 'training/checkpoints'
    }
    config.setdefault('training', {}).update(training_defaults)
    config.setdefault('data', {}).setdefault('path', 'dataset/processed/train.txt')

    run_training(config)

if __name__ == '__main__':
    main()
