
import time
import torch
import yaml
from training.model import GPT, GPTConfig

def benchmark_training(model, config_data):
    """
    Benchmarks the training throughput of the GPT model.
    """
    model.train() # Set to train mode

    model_config = model.config
    vocab_size = model_config.vocab_size
    training_config = config_data['training']

    # Generate dummy data
    batch_size = training_config['batch_size']
    block_size = model_config.block_size
    dummy_input = torch.randint(0, vocab_size, (batch_size, block_size))
    dummy_target = torch.randint(0, vocab_size, (batch_size, block_size))

    # Benchmark settings
    num_steps = 20
    warmup_steps = 5

    # Warmup phase
    for _ in range(warmup_steps):
        _, _, _ = model(dummy_input, dummy_target)

    # Benchmark phase
    start_time = time.time()
    for _ in range(num_steps):
        _, _, _ = model(dummy_input, dummy_target)
    end_time = time.time()

    # Calculate throughput
    total_time = end_time - start_time
    tokens_per_step = batch_size * block_size
    total_tokens = num_steps * tokens_per_step
    tokens_per_second = total_tokens / total_time

    print(f"--- Training Benchmark Results ---")
    print(f"Steps: {num_steps}")
    print(f"Batch Size: {batch_size}")
    print(f"Block Size: {block_size}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Training Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"----------------------------------")
    return tokens_per_second

def benchmark_inference(model, config_data):
    """
    Benchmarks the inference throughput of the GPT model's generate method.
    """
    model.eval() # Set to eval mode

    model_config = model.config

    # Generate a dummy prompt
    prompt = torch.zeros((1, 1), dtype=torch.long)
    max_new_tokens = model_config.block_size - 1 # Generate up to the block size

    # Warmup phase
    _ = model.generate(prompt, max_new_tokens=5)

    # Benchmark phase
    num_runs = 3
    total_tokens_generated = 0

    print(f"\n--- Inference Benchmark (With KV Cache) ---")
    print(f"Generating {max_new_tokens} tokens, {num_runs} times...")

    start_time = time.time()
    for _ in range(num_runs):
        output = model.generate(prompt, max_new_tokens=max_new_tokens)
        tokens_generated = output.size(1) - prompt.size(1)
        total_tokens_generated += tokens_generated
    end_time = time.time()

    total_time = end_time - start_time
    tokens_per_second = total_tokens_generated / total_time

    print(f"Total Tokens Generated: {total_tokens_generated}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Inference Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"-----------------------------------------\n")
    return tokens_per_second


def main(config_path='training/configs/benchmark.yaml'):
    """
    Main function to run the benchmarks.
    """
    # Load configuration from YAML
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Use a dummy vocab_size for benchmark purposes
    vocab_size = 512

    model_config_data = config_data['model']

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=model_config_data['block_size'],
        n_layer=model_config_data['n_layer'],
        n_head=model_config_data['n_head'],
        n_embd=model_config_data['n_embd'],
        dropout=model_config_data['dropout']
    )

    model = GPT(config)

    print(f"--- System Configuration ---")
    print(f"Device: cpu") # Assuming CPU for this benchmark
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"--------------------------\n")

    benchmark_training(model, config_data)
    benchmark_inference(model, config_data)


if __name__ == "__main__":
    main()
