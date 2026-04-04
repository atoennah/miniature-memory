"""
⚡ Bolt: Empirical Performance Benchmark

"In God we trust, all others must bring data."

This script is the scientific ground truth for the model's performance. It isolates
computational throughput and memory efficiency from I/O and preprocessing overhead.
It supports both nested and flat YAML configurations, providing a robust measurement
suite for throughput (tokens/sec), latency (ms/token), and memory (Peak RAM).
"""
import time
import torch
import yaml
import os
import psutil
import argparse
from training.model import GPT, GPTConfig

def run_benchmark(config_path='training/configs/benchmark.yaml'):
    """
    Benchmarks the training throughput, latency, and memory usage of the GPT model.
    """
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    # Load configuration from YAML
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Robust config extraction (handles nested or flat)
    model_cfg = config_data.get('model', config_data)
    train_cfg = config_data.get('training', config_data)

    # Use a dummy vocab_size for benchmark purposes
    vocab_size = 512

    model_cfg = config_data['model']
    train_cfg = config_data['training']

    config = GPTConfig(
        vocab_size=vocab_size,
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
    # Support both flat and nested configuration formats
    model_cfg = config_data.get('model', config_data)
    train_cfg = config_data.get('training', config_data)

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=model_cfg['block_size'],
        n_layer=model_cfg['n_layer'],
        n_head=model_cfg['n_head'],
        n_embd=model_cfg['n_embd'],
        dropout=model_cfg['dropout']
    # Support both nested and flat config structures
    model_config = config_data.get('model', config_data)
    train_config = config_data.get('training', config_data)

    config = GPTConfig(
        vocab_size=vocab_size,
    model_config = config_data['model']
    config = GPTConfig(
        vocab_size=vocab_size,
    training_config = config_data['training']

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=model_cfg['block_size'],
        n_layer=model_cfg['n_layer'],
        n_head=model_cfg['n_head'],
        n_embd=model_cfg['n_embd'],
        dropout=model_cfg['dropout']
        dropout=model_cfg.get('dropout', 0.0)
        block_size=config_data['model']['block_size'],
        n_layer=config_data['model']['n_layer'],
        n_head=config_data['model']['n_head'],
        n_embd=config_data['model']['n_embd'],
        dropout=config_data['model']['dropout']
        block_size=model_config['block_size'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        n_embd=model_config['n_embd'],
        dropout=model_config['dropout']
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(config).to(device)
    model.eval() # Set to eval mode to disable dropout for benchmark

    # Generate dummy data
    batch_size = train_cfg['batch_size']
    block_size = model_cfg['block_size']
    batch_size = train_cfg.get('batch_size', 1)
    block_size = model_cfg['block_size']
    dummy_input = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)
    dummy_target = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)
    batch_size = train_cfg['batch_size']
    block_size = model_cfg['block_size']
    batch_size = config_data['training']['batch_size']
    block_size = config_data['model']['block_size']
    batch_size = train_config['batch_size']
    batch_size = training_config['batch_size']
    batch_size = config_data['training']['batch_size']
    block_size = config_data['model']['block_size']
    batch_size = training_config['batch_size']
    batch_size = config_data['training']['batch_size']
    block_size = model_config['block_size']
    dummy_input = torch.randint(0, vocab_size, (batch_size, block_size))
    dummy_target = torch.randint(0, vocab_size, (batch_size, block_size))

    # Benchmark settings
    num_steps = 20
    warmup_steps = 5

    process = psutil.Process(os.getpid())

    # Warmup phase
    for _ in range(warmup_steps):
        with torch.no_grad():
            _, _, _ = model(dummy_input, dummy_target)
        # The forward pass now returns a third value, the kv_cache, which is not needed for this benchmark.
        _, _, _ = model(dummy_input, dummy_target)

    # Benchmark phase
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()

    for _ in range(num_steps):
        with torch.no_grad():
            _, _, _ = model(dummy_input, dummy_target)

    torch.cuda.synchronize() if device == 'cuda' else None
        _, _, _ = model(dummy_input, dummy_target)
    end_time = time.time()

    peak_memory = process.memory_info().rss / (1024 * 1024) # MB

    # Calculate throughput and latency
    total_time = end_time - start_time
    tokens_per_step = batch_size * block_size
    total_tokens = num_steps * tokens_per_step
    tokens_per_second = total_tokens / total_time
    ms_per_token = (total_time / total_tokens) * 1000

    print(f"--- ⚡ Bolt Benchmark Results ---")
    print(f"Configuration: {config_path}")
    print(f"Device:        {device}")
    print(f"Steps:         {num_steps}")
    print(f"Batch Size:    {batch_size}")
    print(f"Block Size:    {block_size}")
    print(f"Total Tokens:  {total_tokens}")
    print(f"Total Time:    {total_time:.4f} seconds")
    print(f"Throughput:    {tokens_per_second:.2f} tokens/sec")
    print(f"Latency:       {ms_per_token:.4f} ms/token")
    print(f"Peak RAM:      {peak_memory:.2f} MB")
    print(f"--------------------------------")

    return {
        "throughput": tokens_per_second,
        "latency": ms_per_token,
        "memory": peak_memory
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Empirical performance benchmark for the GPT model.")
    parser.add_argument('--config', type=str, default='training/configs/benchmark.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    run_benchmark(args.config)
