
import time
import torch
import yaml
from training.model import GPT, GPTConfig

def run_gen_benchmark(config_path='training/configs/benchmark.yaml'):
    """
    Benchmarks the generation throughput of the GPT model.
    """
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    vocab_size = 512
    model_cfg = config_data['model']

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=model_cfg['block_size'],
        n_layer=model_cfg['n_layer'],
        n_head=model_cfg['n_head'],
        n_embd=model_cfg['n_embd'],
        dropout=model_cfg['dropout']
    )

    model = GPT(config)
    model.eval()

    # Initial context
    idx = torch.zeros((1, 1), dtype=torch.long)
    max_new_tokens = 50

    # Warmup
    _ = model.generate(idx, max_new_tokens=10)

    # Benchmark
    start_time = time.time()
    _ = model.generate(idx, max_new_tokens=max_new_tokens)
    end_time = time.time()

    total_time = end_time - start_time
    tokens_per_second = max_new_tokens / total_time

    print(f"--- Generation Benchmark Results ---")
    print(f"Tokens Generated: {max_new_tokens}")
    print(f"Total Time: {total_time:.4f} seconds")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"------------------------------------")
    return tokens_per_second

if __name__ == "__main__":
    run_gen_benchmark()
