import os
import torch
import argparse
import yaml
from model import GPT, GPTConfig
from typing import Dict, Any, Tuple, List, Callable

# --- Data Loading and Tokenization ---
def get_data(data_path: str) -> Tuple[torch.Tensor, int, Callable[[str], List[int]], Callable[[List[int]], str]]:
    """Reads the training data and creates a simple char-level tokenizer.

    Args:
        data_path (str): The path to the training data file.

    Returns:
        Tuple[torch.Tensor, int, Callable[[str], List[int]], Callable[[List[int]], str]]:
            A tuple containing the data tensor, vocabulary size, encoder, and decoder.
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
        data (torch.Tensor): The full data tensor.
        block_size (int): The context size.
        batch_size (int): The number of sequences in a batch.
        device (str): The device to move the tensors to ('cpu' or 'cuda').

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of input and target tensors.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class Trainer:
    """
    A class to encapsulate the training loop for the NanoGPT model.

    This class handles model initialization, data loading, training,
    and checkpointing, providing a clean interface to run the training process.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Trainer object.

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration
                                      parameters for training, model, and data.
        """
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Create output directory
        os.makedirs(self.config['training']['output_dir'], exist_ok=True)

        # --- Data Loading ---
        self.data, vocab_size, _, _ = get_data(self.config['data']['path'])

        # --- Model Initialization ---
        gpt_config = GPTConfig(
            vocab_size=vocab_size,
            block_size=self.config['model']['block_size'],
            n_layer=self.config['model']['n_layer'],
            n_head=self.config['model']['n_head'],
            n_embd=self.config['model']['n_embd'],
            dropout=self.config['model']['dropout']
        )
        self.model = GPT(gpt_config).to(self.device)

        # --- Optimizer ---
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate'])
        )

    def train(self):
        """
        Runs the main training loop, handles loss calculation, backpropagation,
        and periodic evaluation.
        """
        print("\nStarting training...")
        for step in range(self.config['training']['max_steps']):
            # Get a batch of data
            xb, yb = get_batch(
                self.data,
                self.config['model']['block_size'],
                self.config['training']['batch_size'],
                self.device
            )

            # Evaluate the loss
            _, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if step % self.config['training']['eval_interval'] == 0:
                print(f"Step {step:4d}/{self.config['training']['max_steps']}: Loss: {loss.item():.4f}")

        print("Training finished.")
        self.save_checkpoint()

    def save_checkpoint(self):
        """
        Saves the model's state dictionary to a checkpoint file.
        """
        checkpoint_path = os.path.join(self.config['training']['output_dir'], 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")


def load_config() -> Dict[str, Any]:
    """
    Loads configuration from a YAML file specified by command-line arguments
    and applies necessary defaults.

    This function centralizes configuration management, making the main script
    cleaner and more declarative.

    Returns:
        Dict[str, Any]: The configuration dictionary.
    """
    parser = argparse.ArgumentParser(description="Train a NanoGPT model.")
    parser.add_argument(
        '--config',
        type=str,
        default='training/configs/small.yaml',
        help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply defaults for keys that might be missing from the config file
    # for this simple training loop.
    config.setdefault('training', {})
    config['training'].setdefault('max_steps', 100)
    config['training'].setdefault('eval_interval', 10)
    config['training'].setdefault('output_dir', 'training/checkpoints')

    config.setdefault('data', {})
    config['data'].setdefault('path', 'dataset/processed/train.txt')

    return config


def main(config: Dict[str, Any]):
    """
    The main entry point for the training script.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
    """
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    # 1. Load configuration in a centralized way.
    config = load_config()

    # 2. Instantiate the Trainer and run the training process.
    main(config)
