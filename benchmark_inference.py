
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
    model.eval() # Set to eval mode

    # Generate dummy data for the prompt
    prompt = torch.randint(0, vocab_size, (1, 10)) # Batch size 1, 10 tokens prompt

    # Benchmark settings
    max_new_tokens = 100
    warmup_steps = 2

    # Warmup phase
    print("Warming up...")
    for _ in range(warmup_steps):
        _ = model.generate(prompt, max_new_tokens=max_new_tokens)

    # Benchmark phase
    num_runs = 5
    print(f"Running benchmark ({num_runs} runs)...")
    total_time = 0.0
    for i in range(num_runs):
        start_time = time.time()
        _ = model.generate(prompt, max_new_tokens=max_new_tokens)
        end_time = time.time()
        run_time = end_time - start_time
        total_time += run_time
        print(f"  Run {i+1}/{num_runs}: {max_new_tokens / run_time:.2f} tokens/sec")


    # Calculate average throughput
    average_time = total_time / num_runs
    average_tokens_per_second = max_new_tokens / average_time

    print(f"\n--- Inference Benchmark Results ---")
    print(f"Configuration: {config_path}")
    print(f"Number of Runs: {num_runs}")
    print(f"Generated Tokens per Run: {max_new_tokens}")
    print(f"Average Time: {average_time:.2f} seconds")
    print(f"Average Throughput: {average_tokens_per_second:.2f} tokens/sec")
    print(f"---------------------------------")
    return average_tokens_per_second

if __name__ == "__main__":
    run_inference_benchmark()
