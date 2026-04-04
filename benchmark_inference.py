# [INJECTOR: THE SCIENTIFIC METHOD IN CODE OPTIMIZATION]
#
# This script is not merely a utility; it is an embodiment of the scientific method
# applied to software engineering. Its purpose is to provide a stable, repeatable,
# and empirical foundation for any claims about model performance.
#
# The process is as follows:
# 1.  Observation: We observe that the model's inference speed is a critical
#     performance metric, but we have no way to measure it reliably.
#
# 2.  Hypothesis: We hypothesize that a given change (e.g., adding a KV cache,
#     using a different activation function) will improve inference speed.
#
# 3.  Experiment: We run this script before and after the change, under identical
#     conditions (same hardware, same configuration, same input).
#
# 4.  Verification: We compare the tokens-per-second metric. If the number
#     increases, the hypothesis is supported. If it decreases or stays the same,
#     the hypothesis is falsified.
#
# This script is the arbiter of truth for inference optimization. It replaces
# "I think this is faster" with "I have measured this to be faster," which is the
# cornerstone of a philosophically sound and empirically-driven development process.

"""
A dedicated script for benchmarking the inference performance of the GPT model.

This script measures the tokens-per-second throughput of the model's `generate`
method, providing a consistent way to evaluate the impact of optimizations.
"""
import time
import torch
import argparse
from training.model import GPT, GPTConfig

def run_benchmark(config_path: str, num_tokens: int, prompt: str) -> None:
    """
    Runs the inference benchmark.

    Args:
        config_path: Path to the model configuration YAML file.
        num_tokens: The number of new tokens to generate.
        prompt: The initial prompt to start generation from.
    """
    print("--- Bolt's Inference Benchmark ---")
    print(f"Configuration: {config_path}")
    print(f"Tokens to generate: {num_tokens}")
    print("-" * 30)

    # 1. Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = GPTConfig.from_yaml(config_path)
    model = GPT(config).to(device)

    # [INJECTOR: THE PHILOSOPHY OF A MINIMALIST TOKENIZER]
    #
    # For the purpose of this benchmark, we are not concerned with the semantic
    # correctness of the model's output, but only with its raw token generation
    # speed. Therefore, we use a minimal, ad-hoc character-level tokenizer
    # created on-the-fly from the prompt.
    #
    # This approach is consistent with the project's character-level data
    # preparation and avoids the need for a pre-trained tokenizer, keeping the
    # benchmark self-contained and focused purely on computational throughput.
    chars = sorted(list(set(prompt)))
    vocab_size = len(chars)
    if config.vocab_size < vocab_size:
        print(f"Warning: Model vocab size ({config.vocab_size}) is smaller than prompt vocab ({vocab_size}).")
        print("This may lead to errors. Consider using a prompt with a smaller character set.")

    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]

    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # 2. Warm-up Run
    # The first run can be slower due to CUDA context initialization, etc.
    # We do a warm-up run to get a more accurate measurement.
    print("Running warm-up generation...")
    _ = model.generate(x, max_new_tokens=5)
    print("Warm-up complete.")

    # 3. The Experiment
    print(f"Starting benchmark: generating {num_tokens} tokens...")
    torch.cuda.synchronize() if device == 'cuda' else None

    t0 = time.perf_counter()
    model.generate(x, max_new_tokens=num_tokens)
    torch.cuda.synchronize() if device == 'cuda' else None
    t1 = time.perf_counter()

    # 4. Results & Conclusion
    elapsed_time = t1 - t0
    tokens_per_second = num_tokens / elapsed_time

    print("\n--- Benchmark Results ---")
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print("-" * 30)
    print("Conclusion: The measurement provides a baseline for evaluating future optimizations.")


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark GPT model inference speed.")
    parser.add_argument(
        '--config',
        type=str,
        default='training/configs/small.yaml',
        help='Path to the model configuration YAML file.'
    )
    parser.add_argument(
        '--num_tokens',
        type=int,
        default=200,
        help='Number of new tokens to generate for the benchmark.'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default="Hello, my name is Bolt and I am an AI agent.",
        help='The prompt to start generation from.'
    )
    args = parser.parse_args()

    run_benchmark(args.config, args.num_tokens, args.prompt)


if __name__ == '__main__':
    main()

import time
import torch
import yaml
from training.model import GPT, GPTConfig

def run_inference_benchmark(config_path='training/configs/benchmark.yaml'):
def run_benchmark(config_path='training/configs/benchmark.yaml'):
    """
    Benchmarks the inference throughput of the GPT model's generate method.
    """
    # Load configuration from YAML
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Use a dummy vocab_size for benchmark purposes
    vocab_size = 512

    model_config = config_data['model']
    training_config = config_data['training']

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=model_config['block_size'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        n_embd=model_config['n_embd'],
        dropout=model_config['dropout']
    )

    model = GPT(config)
    model.eval()

    # Generate a dummy input sequence
    start_tokens = torch.zeros((1, 1), dtype=torch.long)

    # Benchmark settings
    num_runs = 5
    max_new_tokens = 100

    # Warmup phase (one generation)
    print("Warming up...")
    _ = model.generate(start_tokens, max_new_tokens=10)
    print("Warmup complete.")

    # Benchmark phase
    print("Running benchmark...")
    start_time = time.time()
    for _ in range(num_runs):
        _ = model.generate(start_tokens, max_new_tokens=max_new_tokens)
    end_time = time.time()
    print("Benchmark complete.")

    # Calculate throughput
    total_time = end_time - start_time
    total_generated_tokens = num_runs * max_new_tokens
    tokens_per_second = total_generated_tokens / total_time

    print(f"--- Inference Benchmark Results ---")
    print(f"Configuration: {config_path}")
    print(f"Number of runs: {num_runs}")
    print(f"Tokens per run: {max_new_tokens}")
    print(f"Total Generated Tokens: {total_generated_tokens}")
    model.eval() # Set to eval mode for generation

    # Generate a dummy starting token
    batch_size = training_config['batch_size']
    start_token = torch.randint(0, vocab_size, (batch_size, 1))

    # Benchmark settings
    tokens_to_generate = 100
    warmup_tokens = 10

    # Warmup phase
    _ = model.generate(start_token, max_new_tokens=warmup_tokens)

    # Benchmark phase
    start_time = time.time()
    _ = model.generate(start_token, max_new_tokens=tokens_to_generate)
    end_time = time.time()

    # Calculate throughput
    total_time = end_time - start_time
    total_tokens = batch_size * tokens_to_generate
    tokens_per_second = total_tokens / total_time

    print(f"--- Inference Benchmark Results ---")
    print(f"Configuration: {config_path}")
    print(f"Batch Size: {batch_size}")
    print(f"Tokens Generated per sample: {tokens_to_generate}")
    print(f"Total Tokens Generated: {total_tokens}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"---------------------------------")
    return tokens_per_second

if __name__ == "__main__":
    run_inference_benchmark()
    run_benchmark()
