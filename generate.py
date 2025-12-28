#!/usr/bin/env python3
import torch
from torch.nn import functional as F
from model import GPT, GPTConfig
from tokenizer import CharacterTokenizer

# --- Constants ---
MODEL_PATH = 'nanogpt_lite.pt'
TOKENIZER_PATH = 'tokenizer.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
START_STRING = "\n"
MAX_TOKENS_TO_GENERATE = 500
TEMPERATURE = 0.8
TOP_P = 0.9

# --- Load Tokenizer ---
tokenizer = CharacterTokenizer.load(TOKENIZER_PATH)
vocab_size = tokenizer.vocab_size
encode = tokenizer.encode
decode = tokenizer.decode

# --- Load Model ---
# We need to know the model's parameters to load it.
# This is a bit of a chicken-and-egg problem. For now, we'll hardcode them.
# A better solution would be to save the config with the model.
config = GPTConfig(vocab_size=vocab_size, block_size=256, n_embd=256, n_head=4, n_layer=7)
model = GPT(config)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- Generation ---
context = torch.tensor(encode(START_STRING), dtype=torch.long, device=DEVICE).unsqueeze(0)

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# --- Run Generation ---
generated_tokens = generate(model, context, max_new_tokens=MAX_TOKENS_TO_GENERATE, temperature=TEMPERATURE)
generated_text = decode(generated_tokens[0].tolist())
print(generated_text)
