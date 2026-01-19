
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

    # Generate a single starting token
    start_token = torch.randint(0, vocab_size, (1, 1))

    # Benchmark settings
    max_new_tokens = 100
    num_runs = 10
    warmup_runs = 3

    # Warmup phase
    print("Running warmup...")
    for _ in range(warmup_runs):
        _ = model.generate(start_token, max_new_tokens=max_new_tokens)
    print("Warmup complete.")

    # Benchmark phase
    print("Running benchmark...")
    throughputs = []
    for i in range(5):
        print(f"  Run {i+1}/5...")
        start_time = time.time()
        for _ in range(num_runs):
            _ = model.generate(start_token, max_new_tokens=max_new_tokens)
        end_time = time.time()

        # Calculate throughput for this run
        total_time = end_time - start_time
        total_tokens_generated = num_runs * max_new_tokens
        tokens_per_second = total_tokens_generated / total_time
        throughputs.append(tokens_per_second)

    # Calculate average throughput
    avg_throughput = sum(throughputs) / len(throughputs)

    print(f"--- Inference Benchmark Results ---")
    print(f"Configuration: {config_path}")
    print(f"Number of benchmark runs: 5")
    print(f"Number of generation runs per benchmark: {num_runs}")
    print(f"Tokens generated per run: {max_new_tokens}")
    print(f"Average Throughput: {avg_throughput:.2f} tokens/sec")
    print(f"Individual run throughputs: {[f'{t:.2f}' for t in throughputs]}")
    print(f"---------------------------------")
    return avg_throughput

if __name__ == "__main__":
    run_inference_benchmark()
