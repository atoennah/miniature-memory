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
