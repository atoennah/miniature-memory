import torch
from training.transformer import GPT, GPTConfig

config = GPTConfig(vocab_size=64, block_size=32, n_layer=1, n_head=1, n_embd=32, dropout=0.0)
model = GPT(config)
x = torch.randint(0, 64, (1, 32))
logits, loss = model(x)
print(f"Logits shape: {logits.shape}")
print(f"Loss: {loss}")

# Test weight tying
print(f"Weight tied: {model.transformer.wte.weight is model.lm_head.weight}")
