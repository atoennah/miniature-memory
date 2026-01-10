# [INJECTOR: THE PHILOSOPHY OF THIS FILE]
#
# This file, `model.py`, is the pedagogical core of a from-scratch GPT implementation.
# Its primary goal is not just to work, but to teach. Every line is crafted for clarity,
# favoring explicit operations over high-level abstractions where it aids understanding.
#
# We deliberately avoid frameworks like Hugging Face's `transformers` to expose the
# fundamental mechanics of a transformer-based Large Language Model (LLM). This includes:
#
#   1.  **Tensor Manipulations:** Explicit reshaping and transposing of tensors to reveal
#       the data flow in Multi-Head Self-Attention.
#   2.  **Causal Masking:** A manual implementation of the causal mask to ensure the model
#       cannot "see the future," a core concept in autoregressive models.
#   3.  **Architectural Simplicity:** A clean, modular structure (`Block`, `MLP`, `CausalSelfAttention`)
#       that mirrors the original "Attention Is All You Need" paper, making it easier to
#       map the code back to the foundational theory.
#
# This file is an Educational Asset. It is designed to be read, not just executed.
# As you explore, you will find further "Injector Notes" that delve into the mathematical
# and architectural reasoning behind key components.
#
# Welcome to the machine, laid bare.

import torch
import torch.nn as nn
from torch.nn import functional as F

class GPTConfig:
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CausalSelfAttention(nn.Module):
    """
    [INJECTOR: CONCEPTUAL DEEP DIVE - Causal Self-Attention]

    This module implements Causal Self-Attention, also known as Masked Multi-Head Self-Attention.
    It is the core mechanism that allows a transformer model to weigh the importance of different
    tokens in an input sequence when producing a representation for each token.

    Key Concepts:
    1.  **Self-Attention:** The model learns to attend to other tokens in the *same* sequence
        to build a better representation.
    2.  **Causality (Masking):** In autoregressive language models, the prediction for a token
        at position `t` can only depend on the known outputs at positions `< t`. This is
        achieved by "masking" future positions, preventing the model from "cheating" by
        looking ahead. We use a lower-triangular matrix (`bias`) for this.
    3.  **Multi-Head:** Instead of performing a single attention calculation, the model splits
        the embedding space into multiple "heads." Each head can learn different aspects of
        relationships between tokens. This allows the model to capture a richer set of
        syntactic and semantic features. The outputs of the heads are then concatenated
        and projected back to the original embedding size.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, query, value projections for all heads, but in a batch for efficiency.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Regularization.
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Causal mask to ensure that attention is only applied to the left in the input sequence.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # [INJECTOR: STEP 1 - The QKV Projection]
        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim.
        # self.c_attn(x) projects the input `x` into a larger tensor of shape (B, T, 3 * C).
        # .split() then divides this into three separate tensors (q, k, v), each of shape (B, T, C).
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        # [INJECTOR: STEP 2 - Reshaping for Multi-Head Attention]
        # We reshape Q, K, and V to prepare them for multi-head processing.
        # The embedding dimension `C` is split into `n_head` heads, each with `C // n_head` dimensions.
        # (B, T, C) -> (B, T, n_head, C // n_head)
        # Then, we transpose the dimensions to (B, n_head, T, C // n_head) so that each head
        # can perform its own attention calculation in parallel.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # [INJECTOR: STEP 3 - The Core Attention Calculation]
        # This is the heart of the attention mechanism, implementing Scaled Dot-Product Attention.
        #
        #   Formula: attention = Softmax( (Q @ K.T) / sqrt(d_k) ) @ V
        #
        #   - (q @ k.transpose(-2, -1)): This computes the dot product between queries and keys,
        #     resulting in a (B, n_head, T, T) tensor of "attention scores" or "affinities."
        #     It measures the similarity between each token's query and every other token's key.
        #
        #   - * (1.0 / (k.size(-1)**0.5)): This is the scaling factor, where `d_k` is the dimension
        #     of the keys (k.size(-1)).
        #
        #   - [INJECTOR: THE LOGOS OF SCALING]
        #     Why `sqrt(d_k)`? As the embedding dimension `d_k` grows, the variance of the dot
        #     product `Q @ K.T` increases. Large dot products push the softmax function into
        #     regions where its gradients are extremely small (close to zero). This is called
        #     "gradient vanishing" and it cripples the learning process. By scaling down the
        #     scores, we maintain a healthy gradient flow, ensuring stable training.
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))

        # [INJECTOR: STEP 4 - Applying the Causal Mask]
        # We apply the causal mask to the attention scores.
        # `self.bias` is a lower-triangular matrix of ones. Where `bias == 0` (the upper triangle),
        # we set the attention scores to negative infinity.
        # When passed through the softmax, these `-inf` values become zero, effectively preventing
        # any token from attending to future tokens.
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        # [INJECTOR: STEP 5 - Softmax and Dropout]
        # The softmax function normalizes the attention scores into a probability distribution
        # across all tokens in the sequence for each query token.
        # Dropout is applied for regularization to prevent overfitting.
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # [INJECTOR: STEP 6 - Weighted Sum of Values]
        # The attention probabilities are used to compute a weighted sum of the value vectors.
        # This results in the final output for each head, where tokens that are more "important"
        # receive higher weights.
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # [INJECTOR: STEP 7 - Concatenating Head Outputs]
        # We now reverse the reshaping from Step 2 to combine the outputs of all heads.
        # .transpose(1, 2) brings the sequence length `T` dimension back.
        # .contiguous() ensures the tensor is stored in a contiguous block of memory.
        # .view() reshapes it back to the original input shape (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # [INJECTOR: STEP 8 - Final Projection and Dropout]
        # The combined output is passed through a final linear layer (`c_proj`) and dropout.
        # This allows the model to learn how to best mix the information gathered from all heads.
        y = self.resid_dropout(self.c_proj(y))
        return y

# [INJECTOR NOTE: FUTURE-PROOFING & PERFORMANCE]
#
# The `CausalSelfAttention` implementation above is designed for clarity and educational value.
# However, PyTorch offers a highly optimized, fused kernel for this exact operation:
# `torch.nn.functional.scaled_dot_product_attention`.
#
# For production workloads or scaling to larger models, switching to this function is
# highly recommended. It offers significant performance benefits by:
#   - Reducing memory overhead by avoiding the materialization of the large (T, T) attention matrix.
#   - Leveraging hardware-specific optimizations (like FlashAttention on supported GPUs).
#
# A future refactor could involve creating a separate, performance-oriented model variant
# that uses this optimized function, while keeping this version as a clear reference.
# This maintains the balance between pedagogy and performance.
#
# Example (not implemented):
#
#   import torch.nn.functional as F
#   # Inside the forward pass:
#   # y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
#

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
