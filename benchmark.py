
import time
import torch
import yaml
from training.model import GPT, GPTConfig

def run_benchmark(config_path='training/configs/benchmark.yaml'):
    """
    Benchmarks the training throughput of the GPT model.
    """
    # Load configuration from YAML
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Use a dummy vocab_size for benchmark purposes
    vocab_size = 512

    # Support both flat and nested config for robustness
    model_cfg = config_data.get('model', config_data)
    train_cfg = config_data.get('training', config_data)

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=model_cfg['block_size'],
        n_layer=model_cfg['n_layer'],
        n_head=model_cfg['n_head'],
        n_embd=model_cfg['n_embd'],
        dropout=model_cfg['dropout']
    )

    model = GPT(config)
    model.eval() # Set to eval mode to disable dropout for benchmark

    # Generate dummy data
    batch_size = train_cfg['batch_size']
    block_size = model_cfg['block_size']
    dummy_input = torch.randint(0, vocab_size, (batch_size, block_size))
    dummy_target = torch.randint(0, vocab_size, (batch_size, block_size))

    # Benchmark settings
    num_steps = 20
    warmup_steps = 5

    # Warmup phase
    for _ in range(warmup_steps):
        _, _, _ = model(dummy_input, dummy_target)

    # Benchmark phase
    start_time = time.time()
    for _ in range(num_steps):
        _, _, _ = model(dummy_input, dummy_target)
    end_time = time.time()

    # Calculate throughput
    total_time = end_time - start_time
    tokens_per_step = batch_size * block_size
    total_tokens = num_steps * tokens_per_step
    tokens_per_second = total_tokens / total_time

    print(f"--- Benchmark Results ---")
    print(f"Configuration: {config_path}")
    print(f"Steps: {num_steps}")
    print(f"Batch Size: {batch_size}")
    print(f"Block Size: {block_size}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"-----------------------")
    return tokens_per_second

if __name__ == "__main__":
    run_benchmark()
