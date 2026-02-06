import sys
import os
import torch
import yaml

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.model import GPT, GPTConfig

def verify_kv_cache():
    print("--- KV-Cache Mathematical Identity Verification ---")

    # Setup model
    config = GPTConfig(
        vocab_size=512,
        block_size=256,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0
    )
    model = GPT(config)
    model.eval()

    # Create a sequence
    seq_len = 10
    idx = torch.randint(0, 512, (1, seq_len))

    # 1. Full forward pass
    print(f"Running full forward pass with sequence length {seq_len}...")
    with torch.no_grad():
        full_logits, _, _ = model(idx)
        # We are interested in the logits for the last token
        last_token_logits_full = full_logits[:, -1, :]

    # 2. Incremental forward pass with KV-cache
    print("Running incremental forward pass with KV-cache...")
    # First, get cache for the first seq_len-1 tokens
    prefix = idx[:, :-1]
    last_token = idx[:, -1:]

    try:
        with torch.no_grad():
            # Initial pass to populate cache
            logits_prefix, _, cache = model(prefix)
            # Incremental pass for the last token
            logits_incremental, _, cache = model(last_token, kv_cache=cache)
            last_token_logits_cached = logits_incremental[:, -1, :]

            # Compare
            diff = torch.abs(last_token_logits_full - last_token_logits_cached).max().item()
            print(f"Max absolute difference: {diff:.2e}")

            if diff < 1e-5:
                print("SUCCESS: Logits are identical!")
            else:
                print("FAILURE: Logits differ!")

    except TypeError as e:
        print(f"FAILED: Model does not yet support KV-cache arguments. Error: {e}")
    except Exception as e:
        print(f"ERROR during verification: {e}")

if __name__ == "__main__":
    verify_kv_cache()
