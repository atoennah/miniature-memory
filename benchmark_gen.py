
import time
import torch
import yaml
from training.model import GPT, GPTConfig

def benchmark_generation(config_path='training/configs/benchmark.yaml'):
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    vocab_size = 512
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=config_data['model']['block_size'],
        n_layer=config_data['model']['n_layer'],
        n_head=config_data['model']['n_head'],
        n_embd=config_data['model']['n_embd'],
        dropout=config_data['model']['dropout']
    )

    model = GPT(config)
    model.eval()

    batch_size = 1
    prompt_length = 10
    max_new_tokens = 50
    idx = torch.randint(0, vocab_size, (batch_size, prompt_length))

    # Warmup
    _ = model.generate(idx, max_new_tokens=5)

    # Benchmark
    start_time = time.time()
    _ = model.generate(idx, max_new_tokens=max_new_tokens)
    end_time = time.time()

    total_time = end_time - start_time
    tokens_per_second = max_new_tokens / total_time

    print(f"--- Generation Benchmark ---")
    print(f"Tokens: {max_new_tokens}")
    print(f"Time: {total_time:.4f}s")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"----------------------------")

if __name__ == "__main__":
    benchmark_generation()
