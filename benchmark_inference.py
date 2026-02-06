
import time
import torch
import yaml
from training.model import GPT, GPTConfig

def benchmark_inference(config_path='training/configs/benchmark.yaml'):
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    vocab_size = 512
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=config_data['model']['block_size'],
        n_layer=config_data['model']['n_layer'],
        n_head=config_data['model']['n_head'],
        n_embd=config_data['model']['n_embd'],
        dropout=0.0
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(config).to(device)
    model.eval()

    max_new_tokens = 200
    start_ids = [0]
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # Warmup
    with torch.no_grad():
        _ = model.generate(x, max_new_tokens=20)

    # Benchmark
    start_time = time.time()
    num_runs = 5
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model.generate(x, max_new_tokens=max_new_tokens)
    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = max_new_tokens * num_runs
    tps = total_tokens / total_time

    print(f"--- Inference Benchmark ---")
    print(f"Tokens Generated: {total_tokens}")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Tokens per second: {tps:.2f}")
    print(f"---------------------------")
    return tps

if __name__ == "__main__":
    benchmark_inference()
