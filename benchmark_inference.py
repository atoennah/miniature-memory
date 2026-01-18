
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

    # In the benchmark config, 'model' is a top-level key.
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
    model.eval() # Set to eval mode for inference

    # Generate a dummy prompt
    batch_size = config_data['training']['batch_size']
    prompt = torch.randint(0, vocab_size, (batch_size, 1))

    # Benchmark settings
    max_new_tokens = 100
    warmup_steps = 2
    num_steps = 5

    # Warmup phase
    print("Running warmup...")
    for _ in range(warmup_steps):
        _ = model.generate(prompt, max_new_tokens=max_new_tokens)
    print("Warmup complete.")

    # Benchmark phase
    print("\nRunning benchmark...")
    start_time = time.time()
    for i in range(num_steps):
        print(f"Step {i+1}/{num_steps}...")
        _ = model.generate(prompt, max_new_tokens=max_new_tokens)
    end_time = time.time()
    print("Benchmark complete.")

    # Calculate throughput
    total_time = end_time - start_time
    tokens_per_step = batch_size * max_new_tokens
    total_tokens_generated = num_steps * tokens_per_step
    tokens_per_second = total_tokens_generated / total_time

    print(f"\n--- Inference Benchmark Results ---")
    print(f"Configuration: {config_path}")
    print(f"Steps: {num_steps}")
    print(f"Batch Size: {batch_size}")
    print(f"New Tokens per Step: {max_new_tokens}")
    print(f"Total Tokens Generated: {total_tokens_generated}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"---------------------------------")
    return tokens_per_second

if __name__ == "__main__":
    run_inference_benchmark()
