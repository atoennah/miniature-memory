
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
        block_size=config_data['block_size'],
        n_layer=config_data['n_layer'],
        n_head=config_data['n_head'],
        n_embd=config_data['n_embd'],
        dropout=config_data['dropout']
    )

    model = GPT(config)
    model.eval() # Set to eval mode

    # Generate dummy prompt
    prompt = torch.randint(0, vocab_size, (1, 10)) # Batch size of 1, 10 tokens long
    max_new_tokens = 100

    # Warmup phase
    _ = model.generate(prompt, max_new_tokens=10)

    # Benchmark phase
    start_time = time.time()
    generated_tokens = model.generate(prompt, max_new_tokens=max_new_tokens)
    end_time = time.time()

    # Calculate throughput
    total_time = end_time - start_time
    num_generated_tokens = len(generated_tokens[0]) - len(prompt[0])
    tokens_per_second = num_generated_tokens / total_time

    print(f"--- Inference Benchmark Results ---")
    print(f"Configuration: {config_path}")
    print(f"Prompt Length: {len(prompt[0])}")
    print(f"Generated Tokens: {num_generated_tokens}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"---------------------------------")
    return tokens_per_second

if __name__ == "__main__":
    run_inference_benchmark()
