import sys
import os
import time
import torch
import yaml

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.model import GPT, GPTConfig

def run_gen_benchmark(config_path='training/configs/benchmark.yaml', max_new_tokens=50):
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    model_config = config_data['model']
    vocab_size = 512

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=model_config['block_size'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        n_embd=model_config['n_embd'],
        dropout=model_config['dropout']
    )

    model = GPT(config)
    model.eval()

    # Initial prompt: a single token
    idx = torch.zeros((1, 1), dtype=torch.long)

    print(f"--- Generation Benchmark ---")
    print(f"Config: {config_path}")
    print(f"Generating {max_new_tokens} tokens...")

    # Warmup
    with torch.no_grad():
        _ = model.generate(idx, max_new_tokens=10)

    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        _ = model.generate(idx, max_new_tokens=max_new_tokens)
    end_time = time.time()

    duration = end_time - start_time
    tps = max_new_tokens / duration

    print(f"Time: {duration:.2f}s")
    print(f"Throughput: {tps:.2f} tokens/sec")
    print(f"----------------------------")
    return tps

if __name__ == "__main__":
    run_gen_benchmark()
