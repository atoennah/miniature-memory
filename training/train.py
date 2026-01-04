import os
import argparse
import sys
import yaml
from typing import Tuple, List, Dict, Any, Callable

import torch
from .model import GPT, GPTConfig

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
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode: Callable[[str], List[int]] = lambda s: [stoi[c] for c in s]
    decode: Callable[[List[int]], str] = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    return data, vocab_size, encode, decode

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
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
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

    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # --- Data ---
    data, vocab_size, _, _ = get_data(config['data']['path'])

    # --- Model ---
    model_config = config['model']
    gpt_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=model_config['block_size'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        n_embd=model_config['n_embd'],
        dropout=model_config['dropout']
    )
    model = GPT(gpt_config).to(device)

    # --- Training ---
    training_config = config['training']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config['learning_rate'])
    )

    print("\nStarting training...")
    for step in range(training_config['max_steps']):
        xb, yb = get_batch(
            data,
            gpt_config.block_size,
            training_config['batch_size'],
            device
        )

        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % training_config['eval_interval'] == 0:
            print(f"Step {step:4d}/{training_config['max_steps']}: Loss: {loss.item():.4f}")

    print("Training finished.")

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
