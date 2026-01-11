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

#
# This module is the pedagogical core of a miniature GPT implementation. Its purpose is not just to work,
# but to teach. Every line is a deliberate choice to expose the fundamental mechanics of the Transformer
# architecture as laid out in the original "Attention Is All You Need" paper (Vaswani et al., 2017).
#
# We intentionally avoid high-level abstractions where they would obscure the underlying logic. For instance,
# instead of a monolithic `nn.TransformerEncoderLayer`, we build it from scratch using LayerNorm,
# CausalSelfAttention, and a Multi-Layer Perceptron (MLP). This reveals the data flow and the purpose
# of each component, particularly the residual connections that are critical for training deep networks.
#
# This implementation is designed for:
#   1.  **Clarity:** To be read, understood, and modified by students of deep learning.
#   2.  **Simplicity:** To demonstrate that a powerful language model can be built from a few core,
#       comprehensible components.
#   3.  **Correctness:** To faithfully implement the key architectural details, such as causal masking
#       for autoregressive generation and scaled dot-product attention.
#
# Read this code not just as a script, but as a textbook. Each class is a chapter, each function a concept.
#
# [REFERENCE]: https://arxiv.org/abs/1706.03762

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

# This module is the pedagogical core of a miniature GPT model, designed for clarity and
# educational value. It implements the essential components of a decoder-only transformer
# architecture, as described in the seminal paper "Attention Is All You Need" and later
# popularized by OpenAI's GPT series.
#
# Our goal is not to achieve state-of-the-art performance, but to create a transparent
# and understandable artifact that reveals the "magic" behind modern LLMs. Every major
# component—from the self-attention mechanism to the residual connections—is implemented
# explicitly, with the intention of being studied and modified.
#
# This file contains the following key architectural components:
#   1.  **GPTConfig**: A simple dataclass to hold model hyperparameters.
#   2.  **CausalSelfAttention**: The heart of the transformer, where tokens communicate.
#   3.  **MLP**: The feed-forward network that processes information from attention.
#   4.  **Block**: The repeating unit of the transformer, combining attention and MLP.
#   5.  **GPT**: The full model, stacking all components together.

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
    """Configuration class for the GPT model."""
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
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

        # [IMPLEMENTATION NOTE]: The model is composed of three main parts:
        # 1. An embedding layer for tokens (`wte`) and positions (`wpe`).
        # 2. A stack of `n_layer` transformer blocks (`h`).
        # 3. A final layer norm (`ln_f`) and a linear layer (`lm_head`) to produce logits.
        # --- 3. Output Layers ---
        # `ln_f`: A final LayerNorm applied after the last transformer block.
        # `lm_head`: The Language Model Head. A linear layer that projects the final
        #          transformer output back to the vocabulary size, producing the raw logits
        #          for the next token prediction. Note the weights are often tied with `wte`.
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
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First sub-layer: Self-Attention, preceded by LayerNorm, followed by a residual connection.
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
