
import time
import torch
import sys
import os

# Ensure the root directory is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.model import GPT, GPTConfig

def benchmark_generation():
    config = GPTConfig(
        vocab_size=50257,
        block_size=512,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0
    )
    model = GPT(config)
    model.eval()

    idx = torch.randint(0, 50257, (1, 1))
    max_new_tokens = 200

    # Warmup
    print("Warming up...")
    _ = model.generate(idx, max_new_tokens=20)

    print(f"Benchmarking generation of {max_new_tokens} tokens...")
    start = time.time()
    _ = model.generate(idx, max_new_tokens=max_new_tokens)
    end = time.time()

    total_time = end - start
    tps = max_new_tokens / total_time
    print(f"Generation throughput: {tps:.2f} tokens/sec")

if __name__ == "__main__":
    benchmark_generation()
