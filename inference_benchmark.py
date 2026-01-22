import time
import torch
from training.model import GPT, GPTConfig

def run_inference_benchmark():
    """
    Benchmarks the inference throughput of the GPT model's generate method.
    """
    # Define a fixed configuration for the benchmark
    config = GPTConfig(
        vocab_size=512,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0
    )

    model = GPT(config)
    model.eval()
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # Generate a dummy prompt
    prompt = torch.randint(0, config.vocab_size, (1, 10)) # Batch size 1, sequence length 10
    max_new_tokens = 100

    # Warmup phase (JIT compilation, etc.)
    print("Warming up...")
    _ = model.generate(prompt, max_new_tokens=5)
    print("Warmup complete.")

    # Benchmark phase
    print("Running benchmark...")
    start_time = time.time()
    generated_tokens = model.generate(prompt, max_new_tokens=max_new_tokens)
    end_time = time.time()

    # Calculate throughput
    total_time = end_time - start_time
    num_generated_tokens = generated_tokens.size(1) - prompt.size(1)

    # Ensure we are measuring the correct number of generated tokens
    assert num_generated_tokens == max_new_tokens

    tokens_per_second = num_generated_tokens / total_time

    print(f"\n--- Inference Benchmark Results ---")
    print(f"Generated {num_generated_tokens} tokens in {total_time:.2f} seconds.")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"-----------------------------------")
    return tokens_per_second

if __name__ == "__main__":
    run_inference_benchmark()
