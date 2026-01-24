
import time
import torch
import yaml
from training.model import GPT, GPTConfig

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

    # --- Inference Benchmark Settings ---
    prompt_size = 10
    max_new_tokens = 50
    num_runs = 10
    warmup_runs = 3

    # Generate a dummy prompt
    # Note: For inference, we typically use a batch size of 1
    prompt = torch.randint(0, vocab_size, (1, prompt_size))

    # --- Warmup Phase ---
    print("Running warmup...")
    for _ in range(warmup_runs):
        _ = model.generate(prompt, max_new_tokens=max_new_tokens)
    print("Warmup complete.")

    # --- Benchmark Phase ---
    print("Running benchmark...")
    start_time = time.time()
    for _ in range(num_runs):
        _ = model.generate(prompt, max_new_tokens=max_new_tokens)
    end_time = time.time()
    print("Benchmark complete.")


    # --- Calculate Throughput ---
    total_time = end_time - start_time
    total_generated_tokens = num_runs * max_new_tokens
    tokens_per_second = total_generated_tokens / total_time

    print(f"\n--- Inference Benchmark Results ---")
    print(f"Configuration: {config_path}")
    print(f"Prompt Size: {prompt_size} tokens")
    print(f"Generated Tokens per Run: {max_new_tokens}")
    print(f"Number of Runs: {num_runs}")
    print(f"Total Generated Tokens: {total_generated_tokens}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Inference Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"-----------------------------------")
    return tokens_per_second

if __name__ == "__main__":
    run_benchmark()
