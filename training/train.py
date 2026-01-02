import os
import torch
import argparse
import yaml
from .model import GPT, GPTConfig
from typing import Dict, Any, Tuple, List

# --- Data Loading and Tokenization ---
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

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    return data, vocab_size, encode, decode

def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a small batch of data of inputs x and targets y.

    Args:
        data: The tensor of encoded text.
        block_size: The context length for predictions.
        batch_size: The number of independent sequences to process in parallel.
        device: The device to move the tensors to ('cpu' or 'cuda').

    Returns:
        A tuple containing:
            - x: The input sequences.
            - y: The target sequences.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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
        )

    def _run_step(self, xb: torch.Tensor, yb: torch.Tensor, step: int) -> None:
        """Performs a single training step."""
        logits, loss = self.model(xb, yb)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if step % self.config['training']['eval_interval'] == 0:
            print(f"Step {step:4d}/{self.config['training']['max_steps']}: Loss: {loss.item():.4f}")

    def _save_checkpoint(self) -> None:
        """Saves the model checkpoint."""
        checkpoint_path = os.path.join(self.config['training']['output_dir'], 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file and applies default values.

    Args:
        config_path: The path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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

    config = load_config(args.config)

    trainer = Trainer(config)
    trainer.run()

if __name__ == '__main__':
    main()
