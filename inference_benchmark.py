
import time
import torch
import yaml
from training.model import GPT, GPTConfig

def run_inference_benchmark():
    """
    Benchmarks the inference throughput (tokens/second) of the GPT model's generate method.
    """
    # [BOLT ⚡]: This configuration is hardcoded to ensure the benchmark is self-contained
    # and does not rely on external files, addressing a code review concern.
    config_data = {
        'block_size': 256,
        'n_layer': 6,
        'n_head': 6,
        'n_embd': 384,
        'dropout': 0.2,
    }
    # Use a dummy vocab_size for benchmark purposes
    vocab_size = 512

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=config_data['block_size'],
        n_layer=config_data['n_layer'],
        n_head=config_data['n_head'],
        n_embd=config_data['n_embd'],
        dropout=config_data['dropout']
    )

    model = GPT(config)
    model.eval() # Set to eval mode to disable dropout for benchmark

    # Benchmark settings
    prompt_size = 1
    max_new_tokens = 100
    num_runs = 10
    warmup_runs = 3

    # Generate a dummy prompt
    dummy_prompt = torch.randint(0, vocab_size, (1, prompt_size))

    print("--- Inference Benchmark ---")
    print(f"Configuration: Hardcoded for benchmark")
    print(f"Generating {max_new_tokens} tokens from a prompt of size {prompt_size}.")
    print(f"Running {num_runs} times...")

    # Warmup phase
    for i in range(warmup_runs):
        print(f"Warmup run {i+1}/{warmup_runs}...")
        _ = model.generate(dummy_prompt, max_new_tokens=max_new_tokens)

    # Benchmark phase
    total_time = 0
    total_tokens_generated = 0

    for i in range(num_runs):
        print(f"Benchmark run {i+1}/{num_runs}...")
        start_time = time.time()
        generated_tokens = model.generate(dummy_prompt, max_new_tokens=max_new_tokens)
        end_time = time.time()

        run_time = end_time - start_time
        tokens_generated = generated_tokens.size(1) - prompt_size

        total_time += run_time
        total_tokens_generated += tokens_generated


    # Calculate average throughput
    average_time_per_run = total_time / num_runs
    average_tokens_per_run = total_tokens_generated / num_runs
    tokens_per_second = average_tokens_per_run / average_time_per_run

    print(f"\n--- Benchmark Results ---")
    print(f"Average time per run: {average_time_per_run:.4f} seconds")
    print(f"Average tokens generated per run: {average_tokens_per_run:.0f}")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"-----------------------")
    return tokens_per_second

if __name__ == "__main__":
    run_inference_benchmark()
