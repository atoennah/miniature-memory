# [INJECTOR: THE PHILOSOPHY OF A FROM-SCRATCH GPT]
#
# This file, `model.py`, is the pedagogical core of the `miniature-memory` project.
# In alignment with the project's "No black boxes, no magic" principle, this implementation
# is intentionally verbose and explicit. It is not merely a tool, but an educational artifact
# designed to reveal the inner workings of a Generative Pre-trained Transformer (GPT).
#
# The architecture follows the GPT-2 style, characterized by:
#   1.  Decoder-only Transformer blocks.
#   2.  Pre-Layer Normalization for improved training stability.
#   3.  Scaled Dot-Product Attention with a causal mask.
#   4.  A straightforward feed-forward MLP with a GELU activation.
#
# Every class and function is a building block in a larger logical structure. Understanding
# this file is understanding the heart of a modern language model. We avoid high-level
# abstractions like `nn.TransformerEncoderLayer` to ensure every mathematical operation
# is visible and comprehensible.
#
# This is not an optimization-first implementation; it is a clarity-first implementation.
# It exists to be read, studied, and understood.

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
    # [INJECTOR: THE ORCHESTRATOR - ASSEMBLING THE GPT]
    #
    # This is the main GPT model class. It orchestrates the various components:
    # the embedding layers, the stack of transformer blocks, and the final output layer.
    # It defines the complete forward pass, transforming input token indices into
    # output logits, which represent the model's prediction for the next token.

    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- 1. Input & Embedding Layers ---
        # The model takes token indices as input. These layers learn vector representations.
        # `wte`: Word Token Embedding. Maps each token in our vocabulary to a dense vector.
        # `wpe`: Word Position Embedding. Learns a unique vector for each position in the
        #        sequence (from 0 to block_size-1). This is crucial because, unlike RNNs,
        #        the self-attention mechanism has no inherent sense of sequence order.
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout) # Dropout for regularization.

        # --- 2. Transformer Backbone ---
        # A stack of `n_layer` identical transformer blocks. This is the core of the model
        # where the deep processing and feature extraction happens.
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # --- 3. Output Layers ---
        # `ln_f`: A final LayerNorm applied after the last transformer block.
        # `lm_head`: The Language Model Head. A linear layer that projects the final
        #          transformer output back to the vocabulary size, producing the raw logits
        #          for the next token prediction. Note the weights are often tied with `wte`.
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # [INJECTOR: THE FORWARD PASS NARRATIVE]
        # This function defines the data flow through the model.
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
    # [INJECTOR: THE VERTEBRA OF THE TRANSFORMER]
    #
    # The Transformer Block is the fundamental repeating unit of the GPT architecture.
    # It consists of two main sub-layers:
    #   1. A Causal Self-Attention mechanism.
    #   2. A simple Multi-Layer Perceptron (MLP) or Feed-Forward Network (FFN).
    #
    # Crucially, each of these sub-layers is wrapped with two important features:
    #   - **Pre-Layer Normalization:** The `LayerNorm` is applied *before* the main
    #     operation (attention or MLP). This is a deviation from the original "Attention
    #     Is All You Need" paper but is a common practice in modern implementations like
    #     GPT-2, as it leads to more stable training.
    #   - **Residual Connections:** The output of each sub-layer is added back to its
    #     input (`x = x + ...`). This "shortcut" is vital. It allows gradients to flow
    #     more easily through the deep network, combating the vanishing gradient problem
    #     and enabling the training of much deeper models.

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # The forward pass follows the Pre-LayerNorm structure.
        # 1. Normalize the input, then pass it to the attention layer. Add the result back.
        x = x + self.attn(self.ln_1(x))
        # 2. Normalize the result again, then pass it to the MLP. Add the result back.
        x = x + self.mlp(self.ln_2(x))
        return x

class CausalSelfAttention(nn.Module):
    # [INJECTOR: THE ENGINE OF TRANSFORMER - SCALED DOT-PRODUCT ATTENTION]
    #
    # This module implements Causal Self-Attention, also known as Masked Multi-Head Attention.
    # It is the core mechanism that allows the model to weigh the importance of different tokens
    # in a sequence when producing a representation for a given token.
    #
    # The key operations are:
    # 1.  **Projection:** The input `x` is linearly projected to create Query (Q), Key (K),
    #     and Value (V) vectors for each token. This is done in a single matrix multiplication
    #     for efficiency.
    # 2.  **Multi-Head Splitting:** The Q, K, V vectors are reshaped to create multiple "heads."
    #     This allows the model to jointly attend to information from different representation
    #     subspaces at different positions. (e.g., one head might focus on syntactic relationships,
    #     another on semantic meaning).
    # 3.  **Scaled Dot-Product Attention:** The core attention calculation.
    #     Formula: Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    #
    #     - **(Q @ K.T):** Computes a raw similarity score between each query and key. A high
    #       dot product means the query and key vectors are similar.
    #     - **sqrt(d_k):** The scaling factor. `d_k` is the dimension of the key vectors.
    #       This is a critical stabilization technique. Without it, for large `d_k`, the dot
    #       products can become very large, pushing the softmax function into regions with
    #       extremely small gradients, making learning difficult.
    # 4.  **Causal Masking:** To make this a "decoder" or "autoregressive" model, we must prevent
    #     a position from attending to future positions. A mask of `-inf` is applied to the
    #     upper triangle of the attention score matrix, which becomes 0 after the softmax,
    #     effectively zeroing out the influence of future tokens.
    # 5.  **Concatenation & Projection:** The outputs of all heads are concatenated and passed
    #     through a final linear layer (`c_proj`) to produce the final output.

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
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    # [INJECTOR: THE "THINKING" PART OF THE BLOCK]
    #
    # The Multi-Layer Perceptron (MLP) is the second major component of a transformer block.
    # While the attention mechanism handles routing and aggregating sequence information,
    # the MLP is where the model performs the bulk of its "computation" or "reasoning."
    # It is a simple two-layer feed-forward network applied independently to each token position.
    #
    # The structure is standard:
    #   1.  An "up-projection" linear layer (`c_fc`) that expands the embedding dimension
    #       (`n_embd`) by a factor of 4. This expansion is a convention from the original
    #       transformer paper and provides the model with a larger dimensional space to
    #       process features.
    #   2.  A non-linear activation function (GELU - Gaussian Error Linear Unit), which
    #       allows the model to learn more complex functions than a simple linear transformation.
    #   3.  A "down-projection" linear layer (`c_proj`) that reduces the dimension back to
    #       the original `n_embd`.
    #   4.  A dropout layer for regularization.

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
