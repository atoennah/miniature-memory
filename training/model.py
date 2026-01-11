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

# This module is not merely a collection of neural network layers; it is a pedagogical tool.
# Its primary purpose is to demystify the inner workings of a Generative Pre-trained Transformer (GPT).
# Every line is crafted for clarity and minimalism, prioritizing understanding over raw performance or feature-completeness.
#
# Core Principles:
# 1. Minimalism: We include only the essential components of a GPT-2 style model. No complex optimizations,
#    no distributed training scaffolding, no labyrinthine configuration files. Just the core architecture.
# 2. Readability: The code is meant to be read, understood, and modified. It serves as a launchpad for
#    experimentation, not a black-box production system.
# 3. Foundational Understanding: By building each component from basic torch.nn modules, we reveal the
#    first principles of self-attention, residual connections, and autoregressive language modeling.
#
# This implementation is heavily inspired by Andrej Karpathy's NanoGPT, a project renowned for its
# educational value. We extend that ethos by injecting verbose commentary to illuminate the "why"
# behind the architectural choices.
#
# Link to original NanoGPT: https://github.com/karpathy/nanogpt
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

# [INJECTOR: THE ARCHITECTURE OF A NANO-GPT]
# This file defines a from-scratch implementation of a Generative Pre-trained Transformer (GPT).
# It is designed to be a pedagogical and minimalistic model, focusing on the core concepts
# that power large language models.
#
# The architecture consists of several key components:
#
# 1.  **Embedding Layer (wte, wpe):**
#     -   `wte` (Word Token Embedding): Maps input vocabulary indices to dense vectors. This is the
#         model's internal representation of a "word" or token.
#     -   `wpe` (Word Position Embedding): Encodes the position of each token in the sequence. Since
#         the transformer architecture itself is permutation-invariant, this is crucial for the model
#         to understand word order.
#
# 2.  **Transformer Blocks (h):**
#     -   This is the core of the model, stacked `n_layer` times. Each block contains two main sub-layers:
#         a.  **Causal Self-Attention:** Allows tokens to look at and gather information from previous
#             tokens in the sequence (but not future ones, hence "causal").
#         b.  **Feed-Forward Network (MLP):** A simple multi-layer perceptron that processes the
#             information gathered by the attention layer. Each token is processed independently.
#
# 3.  **Residual Connections & Layer Normalization:**
#     -   Each sub-layer in a Transformer Block is wrapped with a residual connection (`x = x + sublayer(x)`)
#         and preceded by Layer Normalization. This is critical for enabling the training of deep
#         networks by preventing vanishing/exploding gradients.
#
# 4.  **Final Projection (lm_head):**
#     -   A final linear layer that maps the internal representation of a token back to the vocabulary
#         size, producing the raw logits for the next token prediction.
#
# This implementation follows the GPT-2 architecture but is simplified for educational purposes.

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
    # The `forward` method of this class tells the story of how the model processes
    # a sequence of token IDs (`idx`) to predict the next token.
    #
    #   1.  **Input Embeddings:** The process begins with two embedding layers:
    #       - `wte` (Word Token Embedding): This layer converts each integer token
    #         ID into a dense vector representation. This is the model's first
    #         step in moving from discrete symbols to a continuous, meaningful
    #         space.
    #       - `wpe` (Word Position Embedding): Since the self-attention mechanism
    #         is inherently position-agnostic, we must explicitly provide
    #         positional information. This layer creates a unique vector for each
    #         position in the sequence, which is then added to the token
    #         embedding. This allows the model to understand the order of words.
    #
    #   2.  **Transformer Blocks (`h`):** The summed token and positional embeddings
    #       are then passed through a deep stack of Transformer blocks. This is
    #       the computational core of the model where the deep learning happens.
    #       Each block refines the representations of the tokens by allowing them
    #       to attend to each other and undergo non-linear transformations.
    #
    #   3.  **Final Layer Normalization (`ln_f`):** After the final Transformer
    #       block, one last Layer Normalization is applied. This stabilizes the
    #       final representations before they are projected into the vocabulary space.
    #
    #   4.  **Language Model Head (`lm_head`):** This is a final linear layer that
    #       acts as a "decoder." It takes the high-level token representations
    #       from the Transformer and projects them into a very high-dimensional
    #       space, with one dimension for every word in the vocabulary. These
    #       raw, unnormalized outputs are called "logits."
    #
    #   5.  **Loss Calculation:** If `targets` are provided (during training), the
    #       logits are compared against the true next tokens using the
    #       `cross_entropy` loss function. This function measures how "surprised"
    #       the model was by the true next token. The goal of training is to
    #       minimize this surprise, thereby making the model's predictions more
    #       accurate. A softmax function is implicitly applied to the logits
    #       within `cross_entropy` to convert them into a probability
    #       distribution over the vocabulary.

    def __init__(self, config):
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

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.
    This implementation uses the fused `scaled_dot_product_attention` for efficiency.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # A single linear layer projects the input into Q, K, and V for all heads at once.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # A final projection layer to combine the outputs of all heads.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape # Batch size, Sequence length, Embedding dimensionality

        # 1. Project to Q, K, V for all heads in batch.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 2. Reshape and transpose Q, K, V to prepare for multi-head attention.
        #    The shape changes from (B, T, C) to (B, n_head, T, head_size).
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 3. Perform scaled dot-product attention using the fused kernel.
        #    - `is_causal=True` handles the causal masking automatically.
        #    - Dropout is applied internally if the model is in training mode.
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )

        # 4. Re-assemble all head outputs side-by-side.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5. Apply the final output projection and residual dropout.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First sub-layer: Self-Attention, preceded by LayerNorm, followed by a residual connection.
        x = x + self.attn(self.ln_1(x))
        # Second sub-layer: Feed-Forward, preceded by LayerNorm, followed by a residual connection.
        x = x + self.mlp(self.ln_2(x))
        return x

# [INJECTOR: THE FEED-FORWARD NETWORK]
# This module is a simple Multi-Layer Perceptron (MLP), also known as a Feed-Forward Network (FFN).
# It is applied independently to each token representation.
#
# **Purpose:**
# While the self-attention layer is responsible for gathering and mixing information across the
# sequence, the MLP is responsible for the "computation" or "thinking" on that gathered information.
# It introduces non-linearity and allows the model to learn more complex transformations of the
# token representations.
#
# **Architecture:**
# It consists of two linear layers with a non-linear activation function (GELU) in between.
# The first layer expands the dimensionality of the input (typically by a factor of 4), and the
# second layer projects it back down. This "expansion-compression" structure is a common pattern
# in deep learning, thought to allow the model to learn a richer set of features.

class MLP(nn.Module):
    """A simple multi-layer perceptron."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLP."""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
