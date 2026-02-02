
import time
import torch
import yaml
from training.model import GPT, GPTConfig

def run_gen_benchmark(config_path='training/configs/benchmark.yaml'):
    """
    Benchmarks the generation throughput of the GPT model.
    """
    # Load configuration from YAML
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    vocab_size = 512
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=config_data['model']['block_size'],
        n_layer=config_data['model']['n_layer'],
        n_head=config_data['model']['n_head'],
        n_embd=config_data['model']['n_embd'],
        dropout=config_data['model']['dropout']
    )

    model = GPT(config)
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Prompt
    batch_size = 1 # Generation is usually batch_size 1 for simple cases
    prompt_len = 10
    max_new_tokens = 50
    dummy_input = torch.randint(0, vocab_size, (batch_size, prompt_len)).to(device)

    # Warmup
    print("Warmup...")
    _ = model.generate(dummy_input, max_new_tokens=5, temperature=1.0, top_p=0.9)

    # Benchmark
    print(f"Benchmarking generation of {max_new_tokens} tokens...")
    start_time = time.time()
    _ = model.generate(dummy_input, max_new_tokens=max_new_tokens, temperature=1.0, top_p=0.9)
    end_time = time.time()

    total_time = end_time - start_time
    tokens_per_second = max_new_tokens / total_time

    print(f"--- Generation Benchmark Results ---")
    print(f"Batch Size: {batch_size}")
    print(f"Prompt Length: {prompt_len}")
    print(f"New Tokens: {max_new_tokens}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print(f"-----------------------------------")
    return tokens_per_second

if __name__ == "__main__":
    run_gen_benchmark()
