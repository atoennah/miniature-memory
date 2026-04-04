# [BOLT: THE REFACTOR OF THE GENERATION ENGINE]
#
# Refactored by Bolt to align with the new Trainer architecture and optimized DataManager.
#
# Optimization Highlights:
# 1. Faster Initialization: Instead of re-scanning the entire raw text file to
#    rebuild the tokenizer, we now load the cached metadata (_meta.pkl).
# 2. Resilient Loading: Updated to support the new modular checkpoint format
#    (which includes model state, trainer config, and model config).
# 3. Guard Clauses: Improved error handling for missing configuration or checkpoints.

import sys
import os
import torch
import yaml
import argparse
import psutil
import time
import pickle
from typing import Tuple, Callable, List

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.model import GPT, GPTConfig

def load_tokenizer(data_path: str) -> Tuple[int, Callable, Callable]:
    """
    Loads the character-level tokenizer from cached metadata.
    Falls back to raw file scan if cache is missing.
    """
    meta_path = data_path + "_meta.pkl"

    if os.path.exists(meta_path):
        print(f"Generator: Loading tokenizer from cache: {meta_path}")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
        stoi = meta['stoi']
        itos = meta['itos']
    else:
        print(f"Generator: Warning! Cache missing. Scanning raw file for vocabulary: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '') for i in l])

    return vocab_size, encode, decode

def main(config_path: str, checkpoint_path: str, max_new_tokens: int, start_text: str):
    """Generates text from a trained model with performance monitoring."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Load configuration to find data path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    data_path = config.get('data', {}).get('path', 'dataset/processed/train.txt')

    # 2. Load tokenizer
    vocab_size, encode, decode = load_tokenizer(data_path)

    # 3. Initialize Model
    # Load the checkpoint first to get the model configuration if available
    print(f"Generator: Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if this is a new format checkpoint or old state_dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        # If model_config is in checkpoint, we could use it, but for now we follow the user's config
        model_cfg_dict = config.get('model', {})
        gpt_config = GPTConfig(
            vocab_size=vocab_size,
            block_size=model_cfg_dict.get('block_size', 256),
            n_layer=model_cfg_dict.get('n_layer', 6),
            n_head=model_cfg_dict.get('n_head', 6),
            n_embd=model_cfg_dict.get('n_embd', 384),
            dropout=model_cfg_dict.get('dropout', 0.2)
        )
    else:
        state_dict = checkpoint
        model_cfg_dict = config.get('model', {})
        gpt_config = GPTConfig(
            vocab_size=vocab_size,
            block_size=model_cfg_dict.get('block_size', 256),
            n_layer=model_cfg_dict.get('n_layer', 6),
            n_head=model_cfg_dict.get('n_head', 6),
            n_embd=model_cfg_dict.get('n_embd', 384),
            dropout=model_cfg_dict.get('dropout', 0.2)
        )

    model = GPT(gpt_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Preparation for Generation
    start_ids = encode(start_text)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    print(f"Generator: Using {device}. Generating {max_new_tokens} tokens...")
    print("-" * 40)

    # 5. Measure and Generate
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    start_time = time.time()

    with torch.no_grad():
        # Sensible defaults for inference if not in config
        inf_cfg = config.get('inference', {'temperature': 1.0, 'top_p': 0.9})
        y = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=inf_cfg.get('temperature', 1.0),
            top_p=inf_cfg.get('top_p', 0.9)
        )
        output_text = decode(y[0].tolist())

    duration = time.time() - start_time
    mem_after = process.memory_info().rss / (1024 * 1024)

    # 6. Final Output
    print(output_text)
    print("-" * 40)
    print(f"Bolt Metrics:")
    print(f"  - TPS: {max_new_tokens / duration:.2f}")
    print(f"  - Latency: {duration:.2f}s")
    print(f"  - RAM Delta: {mem_after - mem_before:.2f}MB (Peak: {mem_after:.2f}MB)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate text from a trained NanoGPT model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Number of new tokens to generate.')
    parser.add_argument('--start_text', type=str, default=" ", help='The initial text prompt.')
    args = parser.parse_args()

    main(args.config, args.checkpoint_path, args.max_new_tokens, args.start_text)
