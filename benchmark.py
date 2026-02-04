
import time
import torch
import yaml
import os
from training.model import GPT, GPTConfig

def run_benchmark(config_path='training/configs/benchmark.yaml'):
    """
    Benchmarks the training throughput and generation of the GPT model.
    """
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at '{config_path}'")
        return

    # Load configuration from YAML
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Handle nested or flat config
    if 'model' in config_data:
        m_cfg = config_data['model']
        t_cfg = config_data['training']
    else:
        m_cfg = config_data
        t_cfg = config_data

    # Use a dummy vocab_size for benchmark purposes
    vocab_size = 512

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=m_cfg['block_size'],
        n_layer=m_cfg['n_layer'],
        n_head=m_cfg['n_head'],
        n_embd=m_cfg['n_embd'],
        dropout=m_cfg.get('dropout', 0.0)
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(config).to(device)
    model.eval() # Set to eval mode to disable dropout for benchmark

    # --- Forward Pass Benchmark ---
    batch_size = t_cfg['batch_size']
    block_size = m_cfg['block_size']
    dummy_input = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)
    dummy_target = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)

    # Benchmark settings
    num_steps = 20
    warmup_steps = 5

    # Warmup phase
    for _ in range(warmup_steps):
        with torch.no_grad():
            _, _, _ = model(dummy_input, dummy_target)

    # Benchmark phase
    start_time = time.time()
    for _ in range(num_steps):
        with torch.no_grad():
            _, _, _ = model(dummy_input, dummy_target)
    end_time = time.time()

    # Calculate throughput
    total_time = end_time - start_time
    tokens_per_step = batch_size * block_size
    total_tokens = num_steps * tokens_per_step
    tokens_per_second = total_tokens / total_time

    print(f"--- Forward Pass Benchmark ---")
    print(f"Steps: {num_steps}")
    print(f"Batch Size: {batch_size}")
    print(f"Block Size: {block_size}")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"------------------------------")

    # --- Generation Benchmark ---
    gen_tokens = 200
    num_gen_runs = 3
    start_ids = torch.zeros((1, 1), dtype=torch.long).to(device)

    print(f"--- Generation Benchmark ({gen_tokens} tokens) ---")

    # Warmup
    model.generate(start_ids, max_new_tokens=10)

    gen_times = []
    for i in range(num_gen_runs):
        start_time = time.time()
        model.generate(start_ids, max_new_tokens=gen_tokens)
        gen_times.append(time.time() - start_time)
        print(f"  Run {i+1}: {gen_tokens / gen_times[-1]:.2f} tokens/sec")

    avg_gen_tps = gen_tokens / (sum(gen_times) / len(gen_times))
    print(f"Average Generation Throughput: {avg_gen_tps:.2f} tokens/sec")
    print(f"------------------------------")

    return tokens_per_second, avg_gen_tps

if __name__ == "__main__":
    run_benchmark()
