
import time
import torch
from training.model import GPT, GPTConfig

def run_inference_benchmark(num_tokens=200, prompt_size=10, batch_size=1):
    """
    Benchmarks the inference throughput of the GPT model's generate method.
    """
    # Use a dummy vocab_size and a small model for benchmark purposes
    config = GPTConfig(
        vocab_size=512,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = GPT(config).to(device)
    model.eval()

    # Generate dummy prompt
    prompt = torch.randint(0, config.vocab_size, (batch_size, prompt_size), device=device)

    # Warmup phase
    print("Warming up...")
    _ = model.generate(prompt, max_new_tokens=5)
    print("Warmup complete.")

    # Benchmark phase
    print(f"\nBenchmarking generation of {num_tokens} tokens...")
    start_time = time.time()
    _ = model.generate(prompt, max_new_tokens=num_tokens)
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    # Calculate throughput
    total_time = end_time - start_time
    total_tokens_generated = num_tokens * batch_size
    tokens_per_second = total_tokens_generated / total_time

    print(f"\n--- Inference Benchmark Results ---")
    print(f"Batch Size: {batch_size}")
    print(f"Prompt Size: {prompt_size}")
    print(f"New Tokens Generated: {num_tokens}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"-----------------------------------")
    return tokens_per_second

if __name__ == "__main__":
    run_inference_benchmark()
