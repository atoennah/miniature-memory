# [INJECTOR: THE PHILOSOPHY OF A FROM-SCRATCH GPT]
#
# This file is the pedagogical core of a from-scratch GPT implementation. Its purpose
# is not just to be functional, but to be exceptionally clear and educational. We build
# every core component of the Transformer architecture (as described in "Attention Is All
# You Need") from basic PyTorch primitives.
#
# The hierarchy of the model is as follows:
# 1.  GPT (The full model)
#       - Manages token and positional embeddings.
#       - Contains a stack of Transformer Blocks.
#       - Ends with a layer normalization and a linear layer to produce logits.
#
# 2.  Block (A single Transformer layer)
#       - Encapsulates one Multi-Head Causal Self-Attention module and one MLP.
#       - Employs residual connections ("shortcuts") around both sub-modules, which is
#         critical for training very deep networks.
#
# 3.  CausalSelfAttention (The heart of the Transformer)
#       - Implements Multi-Head Scaled Dot-Product Attention.
#       - A causal mask is applied to ensure that the model is autoregressive (i.e.,
#         predictions for a given token can only depend on previous tokens).
#
# 4.  MLP (Feed-Forward Network)
#       - A simple two-layer feed-forward network with a GELU activation function.
#       - This component is responsible for much of the "thinking" or feature extraction
#         in each Transformer block.
#
# Every design choice here prioritizes clarity over optimization, making it an ideal
# resource for studying the internal mechanics of a language model.

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

# [INJECTOR: RESIDUAL CONNECTIONS & THE MLP]
#
# The `Block` class represents one layer of the Transformer. It consists of two main
# sub-layers: a self-attention mechanism and a feed-forward neural network (MLP).
#
# --- THE POWER OF RESIDUAL CONNECTIONS ---
# The expression `x = x + ...` is a "residual" or "skip" connection. This is a
# foundational innovation from ResNet (He et al., 2015) and is critical for training
# deep neural networks.
#
# Why are they so important?
# 1.  **Combating Vanishing Gradients:** As gradients propagate backward through many
#     layers, they can become vanishingly small. The skip connection provides a direct
#     path for the gradient to flow, ensuring that earlier layers can still learn
#     effectively.
# 2.  **Learning Identity Functions:** It's easier for a layer to learn to output zero
#     (and thus pass the input through unchanged) than it is to learn an identity
#     transformation. This allows the model to dynamically "skip" layers that aren't
#     useful for a particular input, effectively adjusting its own depth.
#
# The Layer Normalization (`ln_1`, `ln_2`) is applied *before* the main transformation
# (a "pre-norm" architecture), which tends to lead to more stable training.
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # The output of the attention layer is added back to the original input.
        x = x + self.attn(self.ln_1(x))
        # The same is done for the MLP layer.
        x = x + self.mlp(self.ln_2(x))
        return x

# [INJECTOR: THE MATHEMATICAL HEART OF THE TRANSFORMER]
#
# This module implements Multi-Head Causal Self-Attention, the component that allows the
# model to weigh the importance of different tokens in the input sequence when producing a
# representation for each token.
#
# --- THEORY: SCALED DOT-PRODUCT ATTENTION ---
# The core of the attention mechanism is the formula:
#
#   Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
#
# Where:
#   - Q (Query): A projection of the input, representing what a token is "looking for."
#   - K (Key): A projection of the input, representing what a token "contains."
#   - V (Value): A projection of the input, representing what a token "offers."
#
# The dot product `Q @ K.T` computes a similarity score between each query and all keys.
# A high score means the key is highly relevant to the query.
#
# --- THE `sqrt(d_k)` SCALING FACTOR ---
# `d_k` is the dimension of the keys. We divide the dot product by `sqrt(d_k)` to prevent
# the values from becoming too large. If the dot products are large, the softmax function
# can saturate (produce very sharp peaks and gradients close to zero), which makes
# training unstable. This scaling ensures that the variance of the dot products remains
# close to 1.
#
# --- CAUSAL MASKING ---
# For language modeling, we need to ensure that the prediction for token `i` can only
# depend on the known outputs at positions less than `i`. This is achieved by "masking"
# future positions. We add negative infinity to the attention scores for all tokens
# that come after the current position, so that when the softmax is applied, their
# probabilities become zero. This is implemented using `torch.tril`.
#
# --- MULTI-HEAD ATTENTION ---
# Instead of performing a single attention calculation, we split the embedding dimension
# into multiple "heads." Each head performs attention independently over a different
# subspace of the embeddings. This allows the model to jointly attend to information
# from different representational subspaces at different positions. The outputs of all
# heads are then concatenated and projected back to the original embedding dimension.
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # B = Batch size, T = Sequence length, C = Embedding dimension
        B, T, C = x.size()

        # --- 1. LINEAR PROJECTION ---
        # Project the input `x` into queries (q), keys (k), and values (v).
        # Instead of three separate linear layers, we use one large layer for efficiency.
        # The output is then split into three parts.
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        # --- 2. RESHAPE FOR MULTI-HEAD ATTENTION ---
        # Reshape q, k, v to prepare them for multi-head processing.
        # The embedding dimension `C` is split into `n_head` heads, each of size `C // n_head`.
        # The dimensions are rearranged so that heads become the batch dimension.
        # Original: (B, T, C) -> Reshaped: (B, n_head, T, head_size)
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)

        # --- 3. SCALED DOT-PRODUCT ATTENTION ---
        # Compute the attention scores ("affinities").
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))

        # --- 4. APPLY CAUSAL MASK ---
        # Mask out future positions to make the attention "causal."
        # The `bias` buffer is a lower-triangular matrix of ones.
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        # --- 5. SOFTMAX & DROPOUT ---
        # Normalize the attention scores into probabilities.
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # --- 6. WEIGHTED AGGREGATION OF VALUES ---
        # The attention scores are used to compute a weighted sum of the values.
        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v

        # --- 7. RESHAPE AND FINAL PROJECTION ---
        # Concatenate the heads back together. `contiguous()` is needed for `view()`.
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection and residual dropout.
        y = self.resid_dropout(self.c_proj(y))
        return y

# [INJECTOR: THE FEED-FORWARD NETWORK (MLP)]
#
# This module is a standard two-layer Multi-Layer Perceptron (MLP), also known as a
# Feed-Forward Network (FFN), which is a component in every Transformer block.
#
# --- ARCHITECTURE ---
# It consists of:
# 1.  An "up-projection" linear layer (`c_fc`) that expands the embedding dimension
#     by a factor of 4.
# 2.  A non-linear activation function (GELU).
# 3.  A "down-projection" linear layer (`c_proj`) that maps it back to the original
#     embedding dimension.
# 4.  A dropout layer for regularization.
#
# --- WHY THE 4X EXPANSION? ---
# The expansion to `4 * n_embd` is a convention established in the original Transformer
# paper ("Attention Is All You Need"). The intuition is that the self-attention layer
# is responsible for routing information between tokens, while the MLP is responsible for
# processing and transforming the content of each token independently. By expanding the
# dimensionality, we create a richer, higher-dimensional space where more complex
# features and relationships can be learned. This is often referred to as the
# "inverted bottleneck" design.
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
