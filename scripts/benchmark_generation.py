import os
import sys
import time
import torch
import yaml
import argparse

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.model import GPT, GPTConfig

def benchmark_generation(config_path='training/configs/small.yaml', max_new_tokens=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load configuration
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Minimal config for benchmarking
    config = GPTConfig(
        vocab_size=100, # Dummy
        block_size=config_data['model']['block_size'],
        n_layer=config_data['model']['n_layer'],
        n_head=config_data['model']['n_head'],
        n_embd=config_data['model']['n_embd'],
        dropout=0.0
    )

    model = GPT(config).to(device)
    model.eval()

    # Input: single token
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    print(f"Benchmarking generation of {max_new_tokens} tokens...")

    # Warmup
    with torch.no_grad():
        _ = model.generate(idx, max_new_tokens=20)

    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        _ = model.generate(idx, max_new_tokens=max_new_tokens)
    end_time = time.time()

    duration = end_time - start_time
    tps = max_new_tokens / duration

    print(f"--- Benchmark Results ---")
    print(f"Tokens: {max_new_tokens}")
    print(f"Time: {duration:.4f}s")
    print(f"Tokens/sec: {tps:.2f}")
    print(f"-------------------------")
    return tps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()
    benchmark_generation(max_new_tokens=args.max_new_tokens)
