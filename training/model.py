# [INJECTOR: THE PHILOSOPHY OF A FROM-SCRATCH GPT]
#
# This module is the pedagogical core of a miniature GPT implementation. It is
# designed to be a transparent, educational asset for understanding the inner
# workings of a Transformer-based language model. Every component is intentionally
# kept explicit and self-contained to reveal the fundamental mechanics of modern
# large language models.
#
# The primary philosophical goals are:
#   1.  **Clarity over Brevity:** The code prioritizes readability and directness
#       over compact, obscure one-liners. The intent is to teach, not to obfuscate.
#   2.  **First Principles Implementation:** We avoid high-level abstractions where
#       they would hide the core logic (e.g., building self-attention from scratch).
#   3.  **Architectural Purity:** This model adheres to the canonical GPT-2/3
#       architecture, including the pre-LayerNorm configuration and causal
#       self-attention, to provide a stable reference point for learning.
#
# This file is not just code; it is a guided tour through the heart of a thinking
# machine. Read it, understand it, and build upon it.
#
# - Bolt ⚡
"""
A minimal, from-scratch GPT model implementation.
Based on Andrej Karpathy's NanoGPT: https://github.com/karpathy/nanogpt
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple

class GPTConfig:
    """Configuration for the GPT model."""
    def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_head: int, n_embd: int, dropout: float):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

class CausalSelfAttention(nn.Module):
    """A causal self-attention module."""

    # [INJECTOR: THE THEORY OF CAUSAL SELF-ATTENTION]
    #
    # This module implements the mathematical core of the Transformer: Scaled
    # Dot-Product Attention, configured for autoregressive (causal) language
    # modeling.
    #
    # ---
    #
    # **The Core Formula: Q, K, V**
    #
    # At its heart, attention is a mechanism for a model to weigh the importance
    # of different tokens in a sequence when producing a representation for a
    # given token. It operates on three vectors derived from the input sequence:
    #
    #   - Q (Query): A representation of the current token, used to "ask" a
    #     question about other tokens.
    #   - K (Key): A representation of other tokens, used to be "queried" by Q.
    #   - V (Value): The actual information content of the other tokens.
    #
    # The similarity between a Query and a Key determines the "attention score."
    # This score is then used to create a weighted sum of the Values, effectively
    # allowing the model to focus on the most relevant parts of the input.
    #
    # ---
    #
    # **Scaled Dot-Product Attention**
    #
    # The specific formula used here is:
    #
    #   Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    #
    # Let's break this down:
    #
    #   1.  `Q @ K.T`: The dot product between the Query and Key vectors. This
    #       computes the raw similarity (or "attention") score between every
    #       pair of tokens.
    #
    #   2.  `/ sqrt(d_k)`: The scaling factor. `d_k` is the dimension of the Key
    #       vectors (head size). This scaling is crucial. Without it, the dot
    #       products can grow very large, pushing the softmax into regions where
    #       its gradients are tiny, killing the learning process. (Reference:
    #       "Attention Is All You Need", Vaswani et al., 2017)
    #
    #   3.  `softmax(...)`: Normalizes the scores into a probability distribution
    #       of "attention weights."
    #
    #   4.  `@ V`: The weighted sum of the Value vectors, producing an output
    #       that blends input tokens based on their relevance.
    #
    # ---
    #
    # **Causal (Masked) Attention**
    #
    # For language modeling, the model cannot "look into the future." This is
    # handled automatically by setting `is_causal=True` in the PyTorch function,
    # which applies a mask to prevent tokens from attending to future tokens.
    #
    # ---
    #
    # **Multi-Head Attention**
    #
    # We split the embedding dimension into multiple "heads." Each head performs
    # its own independent attention calculation on a smaller part of the
    # embedding. The results are then concatenated. This allows the model to
    # jointly attend to information from different representational subspaces.

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the causal self-attention module."""
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape and transpose for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention using PyTorch's fused kernel
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # [INJECTOR: THE ANATOMY OF A TRANSFORMER BLOCK]
    #
    # A Transformer is essentially a stack of these "Block" modules. Each block
    # consists of two main sub-layers:
    # 1. A communication layer: Causal Self-Attention (`attn`). This is where tokens
    #    exchange information with each other.
    # 2. A computation layer: A position-wise Feed-Forward Network (`mlp`). This is where
    #    the model "thinks" about the information it has gathered.
    #
    # Two critical architectural details make this stackable and trainable:
    # a. Residual Connections: The `x + ...` additions. This is a form of shortcut
    #    that allows gradients to flow more easily through the network during backpropagation.
    #    It helps prevent the vanishing gradient problem in deep networks.
    # b. Pre-Layer Normalization: The `ln_1(x)` and `ln_2(x)` calls. Normalizing the input
    #    *before* it enters a sub-layer (as opposed to after) has been shown to stabilize
    #    the training of deep Transformers.

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a transformer block."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """A GPT-style transformer model."""

    # [INJECTOR: THE HIGH-LEVEL ARCHITECTURE OF THE GPT MODEL]
    #
    # This module orchestrates the entire GPT model, assembling all the core
    # components into a coherent, end-to-end language model. It defines the "macro"
    # architecture, detailing the flow of data from input tokens to output logits.
    #
    # ---
    #
    # **The Data Flow: From Integers to Probabilities**
    #
    #   1.  **Input Embeddings (`wte`, `wpe`):** The process begins by converting
    #       integer token IDs into dense vector representations. Positional
    #       embeddings are added to give the model a sense of word order.
    #
    #   2.  **Transformer Blocks (`h`):** The input embeddings are passed through
    #       a deep stack of Transformer blocks. This is the computational core
    #       where tokens are repeatedly processed and refined.
    #
    #   3.  **Final Layer Normalization (`ln_f`):** Stabilizes the final
    #       representations before the final projection.
    #
    #   4.  **Language Model Head (`lm_head`):** A final linear layer that maps the
    #       internal token representation back to the vocabulary size, producing
    #       the raw logits for the next token prediction.

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for the GPT model."""
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # Token and position embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm and language model head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
