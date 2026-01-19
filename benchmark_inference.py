
import time
import torch
import yaml
from training.model import GPT, GPTConfig

def run_benchmark(config_path='training/configs/benchmark.yaml'):
    """
    Benchmarks the inference throughput of the GPT model's generate method.
    """
    # Load configuration from YAML
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Use a dummy vocab_size for benchmark purposes
    # It must be compatible with the model's lm_head
    vocab_size = config_data.get('vocab_size', 512)

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

    # Generate dummy data
    # The starting prompt for generation
    start_idx = torch.zeros((1, 1), dtype=torch.long)

    # Benchmark settings
    max_new_tokens = 100
    num_runs = 5

    throughputs = []
    print("--- Running Inference Benchmark ---")
    for i in range(num_runs):
        start_time = time.time()
        _ = model.generate(start_idx, max_new_tokens=max_new_tokens)
        end_time = time.time()

        total_time = end_time - start_time
        tokens_per_second = max_new_tokens / total_time
        throughputs.append(tokens_per_second)
        print(f"Run {i+1}/{num_runs}: {tokens_per_second:.2f} tokens/sec")

    avg_throughput = sum(throughputs) / len(throughputs)

    print(f"\n--- Benchmark Results ---")
    print(f"Configuration: {config_path}")
    print(f"Generated tokens per run: {max_new_tokens}")
    print(f"Number of runs: {num_runs}")
    print(f"Average Throughput: {avg_throughput:.2f} tokens/sec")
    print(f"-----------------------")
    return avg_throughput

if __name__ == "__main__":
    run_benchmark()
