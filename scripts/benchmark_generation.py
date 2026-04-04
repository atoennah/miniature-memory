
import sys
import os
import time
import torch
import yaml
from typing import List

# Ensure the root directory is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.model import GPT, GPTConfig

def run_benchmark(model, device, num_tokens: int, prompt_length: int):
    idx = torch.randint(0, model.config.vocab_size, (1, prompt_length), device=device)

    # Warmup
    _ = model.generate(idx, max_new_tokens=10)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        _ = model.generate(idx, max_new_tokens=num_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    duration = end_time - start_time
    tokens_per_sec = num_tokens / duration
    return tokens_per_sec

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on {device}")

    config = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0
    )

    model = GPT(config).to(device)
    model.eval()

    test_lengths = [50, 100, 200, 400]
    num_new_tokens = 100

    print(f"{'Prompt Len':<12} | {'New Tokens':<12} | {'Tokens/Sec':<12}")
    print("-" * 42)

    for length in test_lengths:
        tps = run_benchmark(model, device, num_new_tokens, length)
        print(f"{length:<12} | {num_new_tokens:<12} | {tps:<12.2f}")

if __name__ == "__main__":
    main()
