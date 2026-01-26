import time
import torch
from training.model import GPT, GPTConfig

def run_inference_benchmark():
    """
    Benchmarks the inference throughput (tokens/sec) of the GPT model's generate method.
    """
    # Hardcode a small model configuration for a fast, self-contained benchmark
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
    print("Model loaded for benchmarking.")

    # Generate a dummy input prompt
    # B, T (Batch size, sequence length)
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    max_new_tokens = 100

    print("\n--- Running Inference Benchmark ---")
    print(f"Configuration:")
    print(f"  Block size: {config.block_size}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Layers: {config.n_layer}")
    print(f"  Heads: {config.n_head}")
    print(f"  Embedding dim: {config.n_embd}")
    print(f"Prompt length: {prompt.size(1)}")
    print(f"Tokens to generate: {max_new_tokens}")
    print("-----------------------------------")


    # Warmup phase (JIT compilation, etc.)
    print("Running warmup...")
    _ = model.generate(prompt, max_new_tokens=10)
    print("Warmup complete.")

    # Benchmark phase
    print("Starting benchmark...")
    start_time = time.time()
    generated_tokens = model.generate(prompt, max_new_tokens=max_new_tokens)
    end_time = time.time()
    print("Benchmark finished.")

    # Calculate throughput
    total_time = end_time - start_time
    num_generated_tokens = generated_tokens.size(1) - prompt.size(1)

    # Sanity check
    assert num_generated_tokens == max_new_tokens

    tokens_per_second = num_generated_tokens / total_time

    print(f"\n--- Benchmark Results ---")
    print(f"Generated {num_generated_tokens} tokens in {total_time:.2f} seconds.")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"-------------------------")

    return tokens_per_second

if __name__ == "__main__":
    run_inference_benchmark()
