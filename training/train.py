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
import sys
import yaml
from typing import Tuple, List, Dict, Any, Callable

import torch
import time
import torch
import argparse
import torch
import yaml

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
from typing import Dict, Any, Tuple, List
from typing import Dict, Any
import time
from model import GPT, GPTConfig


# --- Data Loading and Tokenization ---

def get_data(data_path: str) -> Tuple[torch.Tensor, int, Callable[[str], List[int]], Callable[[List[int]], str]]:
    """
    Reads training data, creates a character-level tokenizer, and tensorizes the data.

    Args:
        data_path (str): The full path to the training text file.

    Returns:
        A tuple containing:
        - torch.Tensor: The entire dataset as a single tensor of token indices.
        - int: The vocabulary size.
        - Callable[[str], List[int]]: An 'encode' function that maps a string to a list of integers.
        - Callable[[List[int]], str]]: A 'decode' function that maps a list of integers back to a string.
    Reads the training data, creates a simple char-level tokenizer, and returns
    the encoded data and tokenizer functions.

    Args:
        data_path (str): The path to the training data file.

    Returns:
        Tuple[torch.Tensor, int, Callable[[str], List[int]], Callable[[List[int]], str]]:
            A tuple containing:
            - The encoded training data as a PyTorch tensor.
            - The vocabulary size.
            - The 'encode' function for tokenizing strings.
            - The 'decode' function for de-tokenizing lists of integers.
def get_data(data_path: str) -> Tuple[torch.Tensor, int, Any, Any]:
    """Reads training data, creating a character-level tokenizer.

    Args:
        data_path: The path to the training data file.

    Returns:
        A tuple containing:
            - data: A tensor of the encoded text.
            - vocab_size: The size of the vocabulary.
            - encode: A function to encode a string.
            - decode: A function to decode a list of integers.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Simple character-level tokenizer
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode: Callable[[str], List[int]] = lambda s: [stoi[c] for c in s]
    decode: Callable[[List[int]], str] = lambda l: ''.join([itos[i] for i in l])
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
from .data_loader import DataManager
from .trainer import Trainer

def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a small batch of data of inputs x and targets y.

    Args:
        data: The tensor of encoded text.
        block_size: The context length for predictions.
        batch_size: The number of independent sequences to process in parallel.
        device: The device to move the tensors to ('cpu' or 'cuda').

"""
def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
"""
    Generates a small batch of data of inputs x and targets y.
"""
def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
"""
    Generates a random batch of input-target pairs from the dataset.

    Args:
        data (torch.Tensor): The full training dataset as a tensor.
        block_size (int): The context length for predictions.
        batch_size (int): The number of independent sequences in a batch.
        device (str): The device to move the tensors to ('cpu' or 'cuda').

    Returns:
        A tuple containing:
        - torch.Tensor: A batch of input sequences (shape: [batch_size, block_size]).
        - torch.Tensor: A batch of target sequences (shape: [batch_size, block_size]).
    Args:
        data (torch.Tensor): The full training data.
        block_size (int): The context length for predictions.
        batch_size (int): The number of independent sequences to process in parallel.
        device (str): The device to move the tensors to ('cpu' or 'cuda').

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target tensors.

    # Generate random starting indices for the batches
    Returns:
        A tuple containing:
            - x: The input sequences.
            - y: The target sequences.
    """
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
        """
        Initializes the Trainer with a given configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing the model, training,
                                      and data configuration.
        """
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Ensure the output directory for checkpoints exists
        os.makedirs(self.config['training']['output_dir'], exist_ok=True)

        # Load data and tokenizer
        self.data, self.vocab_size, _, _ = get_data(self.config['data']['path'])

        # Initialize the GPT model
        self.model = self._init_model()

        # Initialize the AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate'])
        )

    def _init_model(self) -> GPT:
        """
        Initializes the GPT model based on the configuration.

        Returns:
            GPT: The initialized GPT model.
        """
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
    """A class to encapsulate the training loop for the GPT model."""
    def __init__(self, config: Dict[str, Any]):
        """Initializes the Trainer.

        Args:
            config: A dictionary containing the configuration for the trainer.
        """
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Create output directory
        os.makedirs(self.config['training']['output_dir'], exist_ok=True)

        # Load data
        self.data, self.vocab_size, _, _ = get_data(self.config['data']['path'])

        # Initialize model
        gpt_config = GPTConfig(
            vocab_size=self.vocab_size,
            block_size=self.config['model']['block_size'],
            n_layer=self.config['model']['n_layer'],
            n_head=self.config['model']['n_head'],
            n_embd=self.config['model']['n_embd'],
            dropout=self.config['model']['dropout']
        )
        self.model = GPT(gpt_config).to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate'])
        )

    def run(self) -> None:
        """Runs the main training loop."""
        print("\nStarting training...")
        for step in range(self.config['training']['max_steps']):
            xb, yb = self._get_batch()
            self._run_step(xb, yb, step)

        print("Training finished.")
        self._save_checkpoint()

    def _get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets a batch of data for training."""
        return get_batch(
            self.data,
            self.model.config.block_size,
            self.config['training']['batch_size'],
            self.device
def main(config: Dict[str, Any]) -> None:
    """
    Main function to orchestrate the training pipeline.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
    """
    # Determine the computing device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the DataManager
    data_manager = DataManager(
        data_path=config['data']['path'],
        block_size=config['model']['block_size'],
        batch_size=config['training']['batch_size'],
        device=device
    )

    # --- Trainer ---
    trainer = Trainer(
        config=config,
        vocab_size=data_manager.vocab_size,
        device=device
    )

    # --- Run Training ---
    trainer.train(data_manager)
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate'])
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate'])
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_manager=data_manager,
        device=device,
        max_steps=config['training']['max_steps'],
        eval_interval=config['training']['eval_interval'],
        output_dir=config['training']['output_dir']
    )
    trainer.train()
    # --- Trainer ---
    trainer = Trainer(config, model, optimizer, data_manager)
    trainer.train()
    # --- Trainer ---
    trainer = Trainer(config, data_manager)
    trainer.train()
    )

    # Initialize the Trainer
    trainer = Trainer(config, data_manager)
    # --- Training ---
    trainer = Trainer(config, data_manager)
    trainer.train()
    training_config = config['training']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config['learning_rate'])
    )

    print("\nStarting training...")
    for step in range(training_config['max_steps']):
    t0 = time.time()
    start_time = time.time()
    for step in range(config['training']['max_steps']):
        # Get a batch of data
        xb, yb = get_batch(
            data,
            gpt_config.block_size,
            training_config['batch_size'],
            device
        )

        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
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
    end_time = time.time()
    duration = end_time - start_time
    print(f"Training finished in {duration:.2f} seconds.")

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
    """Loads configuration from a YAML file and applies default values.

    Args:
        config_path: The path to the YAML configuration file.
    # Run the training
    trainer.run()

    Returns:
        A dictionary containing the configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    t1 = time.time()
    elapsed_time = t1 - t0

    total_tokens = config['training']['max_steps'] * config['training']['batch_size'] * gpt_config.block_size
    tokens_per_second = total_tokens / elapsed_time

    print(f"Training finished in {elapsed_time:.2f} seconds.")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    # Apply default values for simplicity in this training script
    config.setdefault('training', {})
    config['training'].setdefault('max_steps', 100)
    config['training'].setdefault('eval_interval', 10)
    config['training'].setdefault('output_dir', 'training/checkpoints')

    config.setdefault('data', {})
    config['data'].setdefault('path', 'dataset/processed/train.txt')

    return config

def main() -> None:
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description="Train a NanoGPT model.")
    parser.add_argument(
        '--config',
        type=str,
        default='training/configs/small.yaml',
        help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
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
    # Load the configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --- Configuration Defaults for Standalone Execution ---
    # These defaults are applied when running the script directly,
    # allowing for quick tests without modifying the main run.py orchestrator.
    # Add some hardcoded values not in the yaml for this simple loop
    # These are defaults for standalone execution of this script.
    # The main `run.py` script will often override these.
    config.setdefault('training', {})
    config['training'].setdefault('max_steps', 100)
    config['training'].setdefault('eval_interval', 10)
    config['training'].setdefault('output_dir', 'training/checkpoints')
    config.setdefault('data', {})
    config['data'].setdefault('path', 'dataset/processed/train.txt')
    # --- Configuration Overrides (for this simple script) ---
    # In a more robust setup, these would be handled by a proper config system
    # or command-line arguments.
    config.setdefault('training', {})['max_steps'] = 100
    config.setdefault('training', {})['eval_interval'] = 10
    config.setdefault('training', {})['output_dir'] = 'training/checkpoints'
    config.setdefault('data', {})['path'] = 'dataset/processed/train.txt'
    # Add block_size to the config if not present, as DataManager needs it.
    config.setdefault('model', {})['block_size'] = 256


    # --- Run the Training ---
    trainer = Trainer(config)
    trainer.train()
    trainer.save_checkpoint()
    config = load_config(args.config)

    trainer = Trainer(config)
    trainer.run()

if __name__ == '__main__':
    main()
    # --- Configuration Defaults ---
    # These values are set to ensure that even a minimal config file will run,
    # especially for standalone execution of this script.
    config.setdefault('training', {})
    config['training'].setdefault('max_steps', 100)
    config['training'].setdefault('eval_interval', 10)
    config['training'].setdefault('output_dir', 'training/checkpoints')

    config.setdefault('data', {})
    config['data'].setdefault('path', 'dataset/processed/train.txt')

    main(config)
