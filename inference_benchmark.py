
import time
import torch
from training.model import GPT, GPTConfig

def run_inference_benchmark():
    """
    Benchmarks the inference throughput of the GPT model's generate method.
    """
    # Define a consistent, small configuration for the benchmark
    config = GPTConfig(
        vocab_size=512,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0
    )

    model = GPT(config)
    model.eval()

    # Generate a dummy prompt
    prompt = torch.randint(0, config.vocab_size, (1, 10)) # Batch size 1, 10 tokens long
    max_new_tokens = 50
    num_runs = 10
    warmup_runs = 3

    print("--- Inference Benchmark ---")
    print(f"Model Config: {config.__dict__}")
    print(f"Prompt length: {prompt.size(1)}, New tokens: {max_new_tokens}")
    print("---------------------------")

    # Warmup phase
    print("Running warmup...")
    for _ in range(warmup_runs):
        _ = model.generate(prompt, max_new_tokens=max_new_tokens)

    # Benchmark phase
    print("Running benchmark...")
    start_time = time.time()
    for _ in range(num_runs):
        _ = model.generate(prompt, max_new_tokens=max_new_tokens)
    end_time = time.time()

    # Calculate throughput
    total_time = end_time - start_time
    total_tokens_generated = num_runs * max_new_tokens
    tokens_per_second = total_tokens_generated / total_time

    print(f"\n--- Benchmark Results ---")
    print(f"Total runs: {num_runs}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total tokens generated: {total_tokens_generated}")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"-----------------------")
    return tokens_per_second

if __name__ == "__main__":
    run_inference_benchmark()
