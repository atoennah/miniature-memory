
import torch
from training.model import GPT, GPTConfig

def verify_kv_cache():
    torch.manual_seed(42)
    config = GPTConfig(vocab_size=100, block_size=128, n_layer=2, n_head=2, n_embd=64, dropout=0.0)
    model = GPT(config)
    model.eval()

    seq_len = 10
    idx = torch.randint(0, 100, (1, seq_len))

    print(f"Verifying KV-cache correctness over {seq_len} steps...")

    # 1. Full pass (baseline)
    with torch.no_grad():
        logits_full, _, _ = model(idx)

    # 2. Incremental pass with KV-cache
    kv_cache = None
    with torch.no_grad():
        for i in range(seq_len):
            curr_idx = idx[:, i:i+1]
            logits_inc, _, kv_cache = model(curr_idx, kv_cache=kv_cache)

            # The incremental logit should match the corresponding index in the full pass
            diff = torch.abs(logits_full[:, i, :] - logits_inc[:, 0, :]).max()
            print(f"  Step {i}: max diff = {diff.item():.2e}")
            if diff > 1e-5:
                print(f"  FAILED at step {i}!")
                return False

    print("Verification SUCCESSFUL: KV-cache output matches full sequence pass.")
    return True

if __name__ == "__main__":
    verify_kv_cache()
