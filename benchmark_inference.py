
import time
import torch
import yaml
from training.model import GPT, GPTConfig

def run_inference_benchmark(config_path='training/configs/benchmark.yaml'):
    """
    Benchmarks the inference throughput of the GPT model's generate method.
    """
    # Load configuration from YAML
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Use a dummy vocab_size for benchmark purposes
    vocab_size = 512

    model_config = config_data['model']
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

    # Generate a dummy input sequence
    start_tokens = torch.zeros((1, 1), dtype=torch.long)

    # Benchmark settings
    num_runs = 5
    max_new_tokens = 100

    # Warmup phase (one generation)
    print("Warming up...")
    _ = model.generate(start_tokens, max_new_tokens=10)
    print("Warmup complete.")

    # Benchmark phase
    print("Running benchmark...")
    start_time = time.time()
    for _ in range(num_runs):
        _ = model.generate(start_tokens, max_new_tokens=max_new_tokens)
    end_time = time.time()
    print("Benchmark complete.")

    # Calculate throughput
    total_time = end_time - start_time
    total_generated_tokens = num_runs * max_new_tokens
    tokens_per_second = total_generated_tokens / total_time

    print(f"--- Inference Benchmark Results ---")
    print(f"Configuration: {config_path}")
    print(f"Number of runs: {num_runs}")
    print(f"Tokens per run: {max_new_tokens}")
    print(f"Total Generated Tokens: {total_generated_tokens}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"---------------------------------")
    return tokens_per_second

if __name__ == "__main__":
    run_inference_benchmark()
