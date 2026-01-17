
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
    run_benchmark()
