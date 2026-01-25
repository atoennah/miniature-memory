
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

    # Generate a dummy prompt
    prompt_length = 10
    batch_size = training_config['batch_size']
    dummy_prompt = torch.randint(0, vocab_size, (batch_size, prompt_length))

    # Benchmark settings
    max_new_tokens = 50
    warmup_runs = 2

    # Warmup phase
    print("Running warmup...")
    for _ in range(warmup_runs):
        model.generate(dummy_prompt, max_new_tokens=max_new_tokens)

    # Benchmark phase
    print("Running benchmark...")
    start_time = time.time()
    generated_tokens = model.generate(dummy_prompt, max_new_tokens=max_new_tokens)
    end_time = time.time()

    # Calculate throughput
    total_time = end_time - start_time
    num_generated_tokens = generated_tokens.size(1) - prompt_length
    tokens_per_second = (batch_size * num_generated_tokens) / total_time

    print(f"--- Inference Benchmark Results ---")
    print(f"Configuration: {config_path}")
    print(f"Batch Size: {batch_size}")
    print(f"Prompt Length: {prompt_length}")
    print(f"Generated Tokens per sample: {num_generated_tokens}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"---------------------------------")
    return tokens_per_second

if __name__ == "__main__":
    run_inference_benchmark()
