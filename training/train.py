import os
import torch
import argparse
import yaml
from typing import Dict, Any, Tuple, List, Callable

from .model import GPT, GPTConfig


# --- Data Loading and Tokenization ---

def get_data(data_path: str) -> Tuple[torch.Tensor, int, Callable[[str], List[int]], Callable[[List[int]], str]]:
    """
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
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Simple character-level tokenizer
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    return data, vocab_size, encode, decode


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a small batch of data of inputs x and targets y.

    Args:
        data (torch.Tensor): The full training data.
        block_size (int): The context length for predictions.
        batch_size (int): The number of independent sequences to process in parallel.
        device (str): The device to move the tensors to ('cpu' or 'cuda').

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target tensors.
    """
    # Generate random starting indices for the batches
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Stack the input sequences into a batch
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Stack the target sequences (shifted by one) into a batch
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a NanoGPT model.")
    parser.add_argument(
        '--config',
        type=str,
        default='training/configs/small.yaml',
        help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --- Configuration Overrides (for this simple script) ---
    # In a more robust setup, these would be handled by a proper config system
    # or command-line arguments.
    config.setdefault('training', {})['max_steps'] = 100
    config.setdefault('training', {})['eval_interval'] = 10
    config.setdefault('training', {})['output_dir'] = 'training/checkpoints'
    config.setdefault('data', {})['path'] = 'dataset/processed/train.txt'

    # --- Run the Training ---
    trainer = Trainer(config)
    trainer.train()
    trainer.save_checkpoint()
