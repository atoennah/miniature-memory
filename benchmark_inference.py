
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

    # Generate dummy data
    batch_size = 1 # Inference is typically done with a single batch
    prompt = torch.randint(0, vocab_size, (batch_size, 10)) # A short prompt
    max_new_tokens = 100

    # Warmup phase
    _ = model.generate(prompt, max_new_tokens=5)

    # Benchmark phase
    start_time = time.time()
    _ = model.generate(prompt, max_new_tokens=max_new_tokens)
    end_time = time.time()

    # Calculate throughput
    total_time = end_time - start_time
    total_tokens = batch_size * max_new_tokens
    tokens_per_second = total_tokens / total_time

    print(f"--- Inference Benchmark Results ---")
    print(f"Configuration: {config_path}")
    print(f"Batch Size: {batch_size}")
    print(f"New Tokens Generated: {max_new_tokens}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"---------------------------------")
    return tokens_per_second

if __name__ == "__main__":
    run_inference_benchmark()
