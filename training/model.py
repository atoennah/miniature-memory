# [INJECTOR: THE PHILOSOPHY OF A FROM-SCRATCH GPT]
# This module is the pedagogical core of a from-scratch GPT implementation.
# Its primary goal is not just to work, but to be understood. It is a verbose,
# heavily-annotated exploration of the key architectural components that make
# up a decoder-only transformer, as described in the seminal paper "Attention Is All You Need."
#
# Every class and function is designed to be as self-contained and explicit as possible,
# favoring clarity and first-principles understanding over high-level abstractions.
# We avoid magic here. The intent is for a developer to read this file and walk away
# with a visceral, mathematical, and architectural understanding of how a Large Language
# Model is built.
#
# Key Concepts Illustrated:
#   1.  **Causal Self-Attention:** The mechanism by which the model focuses on different
#       parts of the input sequence to build context, with a causal mask to prevent it
#       from "seeing the future."
#   2.  **Multi-Head Attention:** The strategy of running the attention mechanism multiple
#       times in parallel to allow the model to attend to different information subspaces.
#   3.  **Transformer Block:** The fundamental repeating unit of the architecture, composed
#       of attention and a feed-forward neural network, connected with residual links
#       and normalization layers.
#   4.  **Positional Encoding:** How the model understands the order of tokens in a sequence,
#       implemented here as a learned embedding.
#
# This is not just code; it is a textbook in executable form.

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
    # [INJECTOR: THE TRANSFORMER BLOCK]
    # This class represents a single block of the transformer architecture. A deep GPT model
    # is composed of many of these blocks stacked on top of each other. Each block contains
    # two main sub-components: a causal self-attention module and a feed-forward neural
    # network (MLP).
    #
    # The architecture follows a pre-normalization style, where Layer Normalization is
    # applied *before* the main operation (attention or MLP). This is a common variant
    # that tends to improve training stability compared to post-normalization.
    #
    # The two `x = x + ...` lines are the residual or "skip" connections. These are
    # absolutely critical for training deep networks. They allow gradients to flow
    # directly through the network, bypassing the transformations in the sub-layers.
    # This mitigates the vanishing gradient problem and allows for much deeper models.

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # [INJECTOR NOTE: RESIDUAL CONNECTION 1]
        # The output of the attention mechanism is added back to the original input `x`.
        # This creates a skip-connection that helps with gradient flow.
        x = x + self.attn(self.ln_1(x))
        # [INJECTOR NOTE: RESIDUAL CONNECTION 2]
        # The output of the MLP is also added back to the result of the previous step.
        x = x + self.mlp(self.ln_2(x))
        return x

class CausalSelfAttention(nn.Module):
    # [INJECTOR: THE HEART OF THE TRANSFORMER - SCALED DOT-PRODUCT ATTENTION]
    # This module implements the causal self-attention mechanism, which is the cornerstone
    # of the transformer architecture. It allows the model to weigh the importance of
    # different tokens in the input sequence when producing a representation for each token.
    #
    # The "causal" aspect is crucial for auto-regressive models like GPT. It ensures that
    # when predicting the next token, the model can only attend to tokens that have come
    # before it, not after. This is enforced by the triangular mask (`self.bias`).
    #
    # The "self-attention" part means the queries, keys, and values all come from the same
    # source sequence (the input `x`). In essence, the sequence is attending to itself.

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # [INJECTOR NOTE]: The `c_attn` layer is a linear projection that creates the
        # Query, Key, and Value matrices for all heads in one shot. This is a common
        # optimization for efficiency. The output dimension is 3 * n_embd because we are
        # creating Q, K, and V, each of size n_embd.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # [INJECTOR NOTE]: `c_proj` is the final linear projection that maps the concatenated
        # outputs of all attention heads back to the original embedding dimension.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # [INJECTOR NOTE: THE CAUSAL MASK]
        # The `bias` buffer is a lower-triangular matrix of ones. This mask is applied
        # to the attention scores before the softmax. By setting the upper-triangular
        # elements to -infinity, we ensure that the model cannot attend to future tokens.
        # `register_buffer` makes this a persistent part of the module, but not a parameter
        # to be trained.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # [INJECTOR NOTE: Q, K, V CALCULATION & MULTI-HEAD SPLIT]
        # 1. Project the input `x` into Q, K, V for all heads in batch.
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        # 2. Reshape and transpose Q, K, V to prepare for multi-head attention.
        #    The dimensions become (B, n_head, T, head_size). This allows each head
        #    to independently process a sub-space of the embedding.
        #    head_size = C // n_head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # [INJECTOR NOTE: SCALED DOT-PRODUCT ATTENTION]
        # 3. Compute attention scores ("affinities"). This is the dot product of Q and K.
        #    Formula: (Q @ K^T) / sqrt(d_k)
        #    The scaling factor `(1.0 / sqrt(k.size(-1)))` is crucial. It prevents the dot
        #    product from growing too large in magnitude, which would push the softmax
        #    into regions with extremely small gradients, making learning unstable.
        #    `k.size(-1)` is the head dimension (`d_k` in the paper).
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
        # 4. Apply the causal mask to the attention scores.
        #    This sets all upper-triangular elements to -infinity, so the softmax
        #    will assign them a probability of 0.
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # 5. Apply softmax to get the attention weights.
        #    The weights represent the importance of each token to every other token.
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # 6. Compute the weighted sum of the values (V). This is the output of the attention.
        #    It is a new representation of the sequence where each token is a blend of
        #    other tokens, weighted by their attention scores.
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # 7. Re-assemble the heads. We concatenate the head outputs back into a single tensor.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Apply the final projection and dropout. This is the output of the multi-head attention block.
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    # [INJECTOR: THE FEED-FORWARD NETWORK]
    # This module is the other major component of a transformer block. It is a simple
    # two-layer Multi-Layer Perceptron (MLP) with a GELU non-linearity in between.
    #
    # Its role is to process the information gathered and aggregated by the self-attention
    # mechanism. While the attention layer is responsible for communication between tokens,
    # the MLP is where the "thinking" or "reasoning" happens on a per-token basis.
    # It provides the representational power to the model.
    #
    # The expansion factor of 4 (from `n_embd` to `4 * n_embd` and back) is a standard
    # architectural choice from the original transformer paper. This intermediate layer
    # is often called the "hidden" or "bottleneck" layer.

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
