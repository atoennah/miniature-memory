
import time
import torch
from training.model import GPT, GPTConfig

def benchmark_generation():
    config = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0
    )
    model = GPT(config)
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    max_new_tokens = 200

    # Warmup
    _ = model.generate(idx, max_new_tokens=10)

    start_time = time.time()
    out = model.generate(idx, max_new_tokens=max_new_tokens)
    end_time = time.time()

    total_time = end_time - start_time
    tokens_per_sec = max_new_tokens / total_time

    print(f"Generation benchmark:")
    print(f"Tokens: {max_new_tokens}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")

if __name__ == "__main__":
    benchmark_generation()
