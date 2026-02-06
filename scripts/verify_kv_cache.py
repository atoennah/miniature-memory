import sys
import os
import torch

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.model import GPT, GPTConfig

def verify():
    print("Verifying KV-Cache mathematical correctness...")
    config = GPTConfig(vocab_size=100, block_size=128, n_layer=2, n_head=2, n_embd=64, dropout=0.0)
    model = GPT(config)
    model.eval()

    # --- Test 1: Simple incremental pass ---
    prompt = torch.randint(0, 100, (1, 10))
    next_token = torch.randint(0, 100, (1, 1))

    with torch.no_grad():
        # Incremental
        _, _, pkvs = model(prompt)
        logits_inc, _, _ = model(next_token, past_key_values=pkvs)

        # Full pass
        full_seq = torch.cat((prompt, next_token), dim=1)
        logits_full, _, _ = model(full_seq)

        diff = torch.abs(logits_full[:, -1, :] - logits_inc[:, -1, :]).max().item()
        print(f"Test 1 (Incremental): Max logit difference = {diff:.2e}")
        if diff > 1e-5:
            print("FAILED Test 1")
            return False

    # --- Test 2: Sequence Boundary (Up to block size) ---
    # Block size is 128. We'll fill it exactly.
    print("Testing sequence boundary (up to block size)...")
    prompt_almost_full = torch.randint(0, 100, (1, 127))
    next_token = torch.randint(0, 100, (1, 1))

    with torch.no_grad():
        # Incremental
        _, _, pkvs = model(prompt_almost_full)
        logits_inc, _, _ = model(next_token, past_key_values=pkvs)

        # Full pass
        full_seq = torch.cat((prompt_almost_full, next_token), dim=1)
        logits_full, _, _ = model(full_seq)

        diff = torch.abs(logits_full[:, -1, :] - logits_inc[:, -1, :]).max().item()
        print(f"Test 2 (Boundary): Max logit difference = {diff:.2e}")
        if diff > 1e-5:
            print("FAILED Test 2")
            return False

    print("Verification SUCCESSFUL! KV-cache is mathematically identical to full forward pass within block_size.")
    print("Note: Beyond block_size, absolute positional embeddings naturally drift due to window sliding.")
    return True

if __name__ == "__main__":
    if not verify():
        sys.exit(1)
