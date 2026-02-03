
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

    # Handle nested or flat config
    if 'model' in config_data:
        m_config = config_data['model']
        t_config = config_data['training']
    else:
        m_config = config_data
        t_config = config_data

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=m_config['block_size'],
        n_layer=m_config['n_layer'],
        n_head=m_config['n_head'],
        n_embd=m_config['n_embd'],
        dropout=m_config.get('dropout', 0.1)
    )

    model = GPT(config)
    model.eval() # Set to eval mode to disable dropout for benchmark

    # Generate dummy data
    batch_size = t_config['batch_size']
    block_size = m_config['block_size']
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

    # Inference Benchmark
    print(f"--- Inference Benchmark (Generation) ---")
    model.eval()
    max_new_tokens = 200
    start_idx = torch.randint(0, vocab_size, (1, 1))

    # Warmup
    _ = model.generate(start_idx, max_new_tokens=5)

    start_time = time.time()
    generated = model.generate(start_idx, max_new_tokens=max_new_tokens)
    end_time = time.time()

    inference_time = end_time - start_time
    inference_tps = max_new_tokens / inference_time

    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Total Time: {inference_time:.2f} seconds")
    print(f"Throughput: {inference_tps:.2f} tokens/sec")
    print(f"-----------------------")

    return tokens_per_second

if __name__ == "__main__":
    run_benchmark()
