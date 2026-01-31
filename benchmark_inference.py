
import time
import torch
import yaml
from training.model import GPT, GPTConfig

def run_inference_benchmark(config_path='training/configs/benchmark.yaml'):
    """
    Benchmarks the inference throughput of the GPT model.
    """
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    vocab_size = 512
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
    model.eval()

    # Inference settings
    batch_size = 1 # Common for single-user inference
    prompt_length = 64
    max_new_tokens = 50

    prompt = torch.randint(0, vocab_size, (batch_size, prompt_length))

    # Warmup
    _ = model.generate(prompt, max_new_tokens=5)

    # Benchmark
    start_time = time.time()
    num_runs = 5
    for _ in range(num_runs):
        _ = model.generate(prompt, max_new_tokens=max_new_tokens)
    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = num_runs * batch_size * max_new_tokens
    tokens_per_second = total_tokens / total_time

    print(f"--- Inference Benchmark Results ---")
    print(f"Tokens Generated: {total_tokens}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"----------------------------------")

if __name__ == "__main__":
    run_inference_benchmark()
