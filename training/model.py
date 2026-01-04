# [INJECTOR: THE LOGOS OF THE TRANSFORMER]
#
# This file is the pedagogical core of a from-scratch GPT implementation.
# Its primary goal is not just to work, but to teach. Every line is intended
# to be as clear and self-contained as possible, demystifying the "magic"
# behind Large Language Models.
#
# PHILOSOPHY:
# 1.  **Clarity over Brevity:** Code is written to be read and understood by a
#     human first, and a machine second. Complex one-liners are avoided in
#     favor of multi-line, commented explanations.
# 2.  **First Principles:** We avoid high-level abstractions where they obscure
#     the underlying mechanics. For example, the attention mechanism is built
#     manually to show the tensor transformations explicitly.
# 3.  **Educational Asset:** This is a living document. It serves as a reference
#     for understanding the fundamental building blocks of a transformer,
#     including:
#       - Scaled Dot-Product Attention
#       - Multi-Head Attention
#       - Residual Connections & Layer Normalization
#       - Positional Encodings
#       - The Decoder-Only Architecture
#
# By studying this file, one should be able to grasp the core concepts
# necessary to build and train their own language models.

import torch
import torch.nn as nn
from torch.nn import functional as F


class GPTConfig:
    """Configuration class for the GPT model."""
    def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_head: int, n_embd: int, dropout: float):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

class GPT(nn.Module):
    """
    The main GPT model class.

    This class orchestrates the entire transformer architecture, combining the
    embedding layers, the stack of transformer blocks, and the final output
    layer.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token and positional embedding layers.
        # `wte` learns vector representations for each token in the vocabulary.
        # `wpe` learns vector representations for each position in the sequence.
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # The stack of transformer blocks. This is the core of the model.
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final layer normalization and the output linear layer.
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
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
    """
    A single block of the transformer.

    Each block consists of a causal self-attention module followed by a
    feed-forward neural network (MLP). Residual connections ("shortcuts") and
    layer normalization are applied around each of these two sub-layers.
    This is a standard "pre-norm" architecture, where LayerNorm is applied
    before the main operation.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The output of the attention layer is added to the original input (residual connection).
        x = x + self.attn(self.ln_1(x))
        # The output of the MLP is added to the result of the attention layer (another residual connection).
        x = x + self.mlp(self.ln_2(x))
        return x

# [INJECTOR: THE MATHEMATICS OF ATTENTION]
#
# At its core, "attention" is a mechanism for a model to weigh the importance
# of different parts of an input sequence. When the model processes a word,
# self-attention allows it to look at other words in the input sequence for
# clues that can help lead to a better encoding for this word.
#
# This module implements Causal Self-Attention, which has two key properties:
# 1.  **Self-Attention:** The queries (Q), keys (K), and values (V) are all
#     derived from the same input sequence (x). It's the sequence attending
#     to itself.
# 2.  **Causal (or Masked):** During training, to predict the next token, the
#     model should only have access to the tokens that came before it, not
#     the tokens that come after. The causal mask (the `bias` buffer) ensures
#     this by zeroing out attention scores for future positions.
#
# The formula for Scaled Dot-Product Attention is:
#
#   Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
#
# Breaking it down:
#   - Q @ K.T:      Calculates a raw "similarity" score between each query and
#                   every key. A high dot product means the vectors are similar.
#   - / sqrt(d_k):  This is the scaling factor. It prevents the dot products
#                   from becoming too large, which would push the softmax into
#                   regions with very small gradients, making learning difficult.
#                   `d_k` is the dimensionality of the key vectors.
#   - softmax(...): Normalizes the scores into a probability distribution.
#   - @ V:          A weighted sum of the value vectors, where the weights are
#                   the attention probabilities. Tokens that are more "important"
#                   (higher attention score) contribute more to the output.
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # A single linear layer projects the input into Q, K, and V matrices.
        # This is an optimization: instead of 3 separate linear layers, we do
        # one and then split the result. It's computationally equivalent and
        # often faster.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # The output projection layer. After computing the weighted sum of
        # values, we project the result back into the embedding dimension.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # The causal mask. This is a lower-triangular matrix of ones.
        # When applied, it ensures that a position `i` can only attend to
        # positions `j <= i`. The `register_buffer` makes this a persistent
        # part of the module, but not a parameter to be trained.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # Input shape: (Batch Size, Sequence Length, Embedding Dimension)
        B, T, C = x.size()

        # [INJECTOR NOTE: TENSOR TRANSFORMATION FOR MULTI-HEAD ATTENTION]
        # 1.  Calculate Q, K, V for all heads in batch.
        #     The input `x` is passed through one large linear layer `c_attn`,
        #     and the result is split into three chunks for Q, K, and V.
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        # 2.  Reshape and transpose for multi-head computation.
        #     We transform the tensors from (B, T, C) into
        #     (B, n_head, T, head_size) to perform attention calculations
        #     in parallel for each head. `head_size` is `C // n_head`.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 3.  Perform scaled dot-product attention.
        #     `q @ k.transpose(-2, -1)` results in (B, nh, T, T), the raw
        #     attention scores.
        #     The scaling factor `1.0 / sqrt(k.size(-1))` is applied.
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))

        # 4.  Apply the causal mask.
        #     The `masked_fill` operation replaces all values where the mask
        #     is 0 with negative infinity. This is done before the softmax
        #     so that these positions get a probability of 0.
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        # 5.  Normalize with softmax and apply dropout.
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 6.  Compute the weighted sum of values.
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        # 7.  Reshape and combine the heads.
        #     The `contiguous()` call is needed to ensure the tensor is stored
        #     in a contiguous block of memory before the `view` operation.
        #     This concatenates the head outputs back into a single tensor.
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        # 8.  Apply the final output projection and dropout.
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """
    The Multi-Layer Perceptron (or Feed-Forward Network) part of the transformer block.

    This is a simple two-layer neural network with a GELU activation function in
    between. It is applied to each position (token) independently. The first
    layer expands the embedding dimension by a factor of 4, and the second
    layer projects it back down.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
