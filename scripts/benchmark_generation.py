import sys
import os
import torch
import time
import yaml
import argparse
import psutil

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.model import GPT, GPTConfig

def get_vocab_size(data_path):
    """Gets the vocabulary size from the training data."""
    if not os.path.exists(data_path):
        return 256 # Fallback
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    return len(chars)

def main():
    parser = argparse.ArgumentParser(description="Benchmark GPT generation speed.")
    parser.add_argument('--config', type=str, default='training/configs/benchmark.yaml', help='Path to the config file.')
    parser.add_argument('--checkpoint', type=str, default='out-benchmark/model.pt', help='Path to the checkpoint.')
    parser.add_argument('--tokens', type=int, default=100, help='Number of tokens to generate.')
    parser.add_argument('--prompt', type=str, default=' ', help='Initial prompt.')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        return

    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    vocab_size = get_vocab_size(config_data['data']['path'])

    gpt_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=config_data['model']['block_size'],
        n_layer=config_data['model']['n_layer'],
        n_head=config_data['model']['n_head'],
        n_embd=config_data['model']['n_embd'],
        dropout=0.0
    )

    model = GPT(gpt_config)
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print("Warning: Checkpoint not found, using random weights.")

    model.to(device)
    model.eval()

    # Simple character encoding for benchmark
    if os.path.exists(config_data['data']['path']):
        with open(config_data['data']['path'], 'r', encoding='utf-8') as f:
            chars = sorted(list(set(f.read())))
        stoi = {ch: i for i, ch in enumerate(chars)}
        encode = lambda s: [stoi.get(c, 0) for c in s]
    else:
        encode = lambda s: [0] * len(s)

    start_ids = encode(args.prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    print(f"Benchmarking generation of {args.tokens} tokens (Context size: {config_data['model']['block_size']})...")

    process = psutil.Process(os.getpid())
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    start_time = time.time()
    mem_before = process.memory_info().rss / (1024 * 1024)

    with torch.no_grad():
        _ = model.generate(x, max_new_tokens=args.tokens, temperature=1.0, top_p=0.9)

    if device == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()
    mem_after = process.memory_info().rss / (1024 * 1024)

    duration = end_time - start_time
    tps = args.tokens / duration

    print("-" * 30)
    print(f"Performance Metrics:")
    print(f"  - Generation Time: {duration:.4f}s")
    print(f"  - Throughput:      {tps:.2f} tokens/sec")
    print(f"  - Peak RAM Usage:  {mem_after:.2f} MB")
    print(f"  - RAM Delta:       {mem_after - mem_before:.2f} MB")
    print("-" * 30)

if __name__ == '__main__':
    main()
