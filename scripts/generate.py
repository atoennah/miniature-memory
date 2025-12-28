import sys
import os
# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import yaml
import argparse
import psutil
import time
from training.model import GPT, GPTConfig

def get_tokenizer(data_path):
    """Creates the same character-level tokenizer from the training data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    return vocab_size, encode, decode

def main(config):
    """Generates text from a trained model and measures resource usage."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Load Model and Tokenizer ---
    vocab_size, encode, decode = get_tokenizer(config['data']['path'])

    gpt_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=config['model']['block_size'],
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        n_embd=config['model']['n_embd'],
        dropout=config['model']['dropout']
    )

    model = GPT(gpt_config)
    model.load_state_dict(torch.load(config['generate']['checkpoint_path'], map_location=device))
    model.to(device)
    model.eval()

    # --- Generation ---
    start_text = config['generate'].get('start_text', ' ')
    start_ids = encode(start_text)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    print(f"Generating {config['generate']['max_new_tokens']} tokens from prompt: '{start_text}'")
    print("-" * 30)

    # --- Measurement ---
    process = psutil.Process(os.getpid())
    start_time = time.time()
    mem_before = process.memory_info().rss / (1024 * 1024) # in MB

    # Generate text
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=config['generate']['max_new_tokens'])
        output_text = decode(y[0].tolist())

    mem_after = process.memory_info().rss / (1024 * 1024) # in MB
    end_time = time.time()

    # --- Output ---
    print(output_text)
    print("-" * 30)

    # --- Performance Metrics ---
    duration = end_time - start_time
    tokens_per_sec = config['generate']['max_new_tokens'] / duration if duration > 0 else float('inf')

    print(f"Performance Metrics:")
    print(f"  - Peak RAM Usage: {mem_after:.2f} MB")
    print(f"  - Initial RAM:    {mem_before:.2f} MB")
    print(f"  - Generation Time:  {duration:.2f} seconds")
    print(f"  - Tokens per Second: {tokens_per_sec:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate text from a trained NanoGPT model.")
    parser.add_argument(
        '--config',
        type=str,
        default='training/configs/small.yaml',
        help='Path to the YAML configuration file used for training.'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=100,
        help='Number of new tokens to generate.'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Add generation-specific config values
    config.setdefault('generate', {})['checkpoint_path'] = 'training/checkpoints/model.pt'
    config.setdefault('generate', {})['max_new_tokens'] = args.max_new_tokens
    config.setdefault('data', {})['path'] = 'dataset/processed/train.txt'

    main(config)
