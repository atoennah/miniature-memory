# [INJECTOR: THE PHILOSOPHY OF A FROM-SCRATCH GPT]
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
    """Configuration class for the GPT model."""
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
    """
    The main GPT model class.

    This class orchestrates the entire transformer architecture, combining the
    embedding layers, the stack of transformer blocks, and the final output
    layer.
    """
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
    block_size: int = 256
    vocab_size: int = 65 # Default: char-level for English text
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2

class MultiHeadAttention(nn.Module):
    # [INJECTOR: MERGED IMPLEMENTATION - ATTENTION LOGOS & PARALLEL REALITIES]
    #
    # This module combines the logic of single-head attention and multi-head attention
    # into a single, optimized implementation that leverages PyTorch's fused kernels.
    #
    # --- The Logos of Self-Attention ---
    # The core formula remains: Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    # Q (Query): What I am looking for.
    # K (Key):   What I contain.
    # V (Value): What I will give you.
    #
    # --- The "Parallel Realities" of Multi-Head Attention ---
    # Instead of iterating through separate `Head` modules, we perform the projections for
    # all heads simultaneously using a single `c_attn` linear layer. This is highly efficient.
    # The output of this layer (3 * n_embd) is then split and reshaped to create the Q, K, V
    # tensors for all heads at once.
    #
    # --- Fused Kernel Optimization ---
    # The entire sequence of scaling, masking, softmax, and value-weighting is performed
    # in a single call to `torch.nn.functional.scaled_dot_product_attention`. This function
    # is highly optimized to use hardware-specific kernels like FlashAttention when available.
    # The `is_causal=True` argument replaces the manual lower-triangular mask (`tril`),
    # ensuring that the model remains autoregressive.
    """Multiple heads of self-attention in parallel"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Special scaling for residual connections, per NanoGPT
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # Regularization
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # Flash attention is implemented in PyTorch 2.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

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
        # A single linear layer projects the input into Q, K, and V for all heads at once.
        # This is more efficient than creating separate linear layers for each head.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # A final projection layer to combine the outputs of all heads.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd


    def forward(self, x):
        B, T, C = x.shape # Batch size, Sequence length, Embedding dimensionality (n_embd)

        # 1. Project to Q, K, V for all heads in batch.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 2. Reshape and transpose Q, K, V to prepare for multi-head attention.
        #    The shape changes from (B, T, C) to (B, n_head, T, head_size).
        #    This brings the head dimension forward, allowing each head to perform attention independently.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 3. Perform scaled dot-product attention using the fused kernel.
        #    - `is_causal=True` handles the causal masking automatically.
        #    - Dropout is applied internally if the model is in training mode.
        #    - The scaling by 1/sqrt(head_size) is also handled internally.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        # 4. Re-assemble all head outputs side-by-side.
        #    The transpose and contiguous().view() operations reverse the reshaping from step 2,
        #    concatenating the head outputs to get a final shape of (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5. Apply the final output projection.
        y = self.dropout(self.c_proj(y))
        return y
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        # is_causal=True will automatically apply the lower-triangular mask
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)

class Block(nn.Module):
    # [INJECTOR: THE ANATOMY OF A TRANSFORMER BLOCK]
    #
    # A Transformer is essentially a stack of these "Block" modules. Each block
    # consists of two main sub-layers:
    # 1. A communication layer: Multi-Head Self-Attention (`sa`). This is where tokens
    #    exchange information with each other.
    # 2. A computation layer: A position-wise Feed-Forward Network (`ffwd`). This is where
    #    the model "thinks" about the information it has gathered.
    #
    # Two critical architectural details make this stackable and trainable:
    # a. Residual Connections: The `x + ...` additions. This is a form of shortcut
    #    that allows gradients to flow more easily through the network during backpropagation.
    #    It helps prevent the vanishing gradient problem in deep networks by allowing the
    #    model to learn an identity function if a layer is not useful.
    # b. Pre-Layer Normalization: The `ln1(x)` and `ln2(x)` calls. Normalizing the input
    #    *before* it enters a sub-layer (as opposed to after) has been shown to stabilize
    #    the training of deep Transformers. It keeps the activations well-behaved throughout
    #    the network.
    """Transformer block: communication followed by computation"""
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for the GPT model.

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

    # TODO [SCALING]: To scale this model to billions of parameters, several architectural
    # changes would be necessary:
    #   1.  **Distributed Training**: The model would need to be wrapped in `DistributedDataParallel` (DDP)
    #       for multi-GPU training.
    #   2.  **Model Parallelism**: For very large models that don't fit on a single GPU, techniques
    #       like Tensor Parallelism (splitting individual layers across GPUs) and Pipeline Parallelism
    #       (placing different layers on different GPUs) would be required. Libraries like
    #       DeepSpeed or FSDP (Fully Sharded Data Parallel) in PyTorch handle this complexity.
    #   3.  **Activation Checkpointing**: To save memory, activation checkpointing (or gradient
    #       checkpointing) can be used. This trades compute for memory by not storing all
    #       activations in the forward pass and recomputing them during the backward pass.

    def forward(self, idx, targets=None):
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        Args:
            idx: A tensor of shape (B, T) containing the input token indices.
            targets: An optional tensor of shape (B, T) containing the target token indices.

        Returns:
            A tuple containing:
                - logits: A tensor of shape (B, T, vocab_size) containing the model's predictions.
                - loss: An optional tensor containing the cross-entropy loss.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # Token and position embeddings
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    def forward(self, x):
        # First sub-layer: Self-Attention, preceded by LayerNorm, followed by a residual connection.
        x = x + self.sa(self.ln1(x))
        # Second sub-layer: Feed-Forward, preceded by LayerNorm, followed by a residual connection.
        x = x + self.ffwd(self.ln2(x))
        return x
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.h:
            x = block(x)

        # Final layer norm and language model head
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Takes a conditioning sequence of indices (LongTensor of shape (b,t)) and completes
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :]
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
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
# [INJECTOR: THE TRANSFORMER BLOCK]
# This module represents one layer of the Transformer. It is composed of two main sub-layers:
# 1. A Causal Self-Attention mechanism.
# 2. A simple Multi-Layer Perceptron (MLP) or Feed-Forward Network.
#
# **Residual Connections (`x = x + ...`):**
# The addition operation (`+`) implements a residual or "skip" connection. This is a critical
# architectural innovation that allows gradients to flow directly through the network, bypassing
# the non-linear transformations of the sub-layers. It enables the training of much deeper
# models by mitigating the vanishing gradient problem.
#
# **Pre-Layer Normalization:**
# We apply Layer Normalization (`ln_1`, `ln_2`) *before* each sub-layer, which is a modern
# convention (GPT-2 style) that tends to improve training stability compared to applying it after.

class Block(nn.Module):
    # [INJECTOR: THE BUILDING BLOCK OF THE TRANSFORMER]
    # This module represents a single block of a transformer. It consists of two main sub-layers:
    # 1. A Causal Self-Attention layer.
    # 2. A Feed-Forward Network (MLP).
    #
    # Each of these sub-layers has a residual connection ("skip connection") around it, followed
    # by a Layer Normalization. This is a "Pre-LayerNorm" architecture, where the normalization
    # is applied *before* the main operation (attention or MLP).
    """
    A single block of the transformer.

    Each block consists of a causal self-attention module followed by a
    feed-forward neural network (MLP). Residual connections ("shortcuts") and
    layer normalization are applied around each of these two sub-layers.
    This is a standard "pre-norm" architecture, where LayerNorm is applied
    before the main operation.
    """
    def __init__(self, config: GPTConfig):
    """A single transformer block."""
    def __init__(self, config: GPTConfig):
    # [INJECTOR: THE ARCHITECTURE OF A TRANSFORMER BLOCK]
    #
    # This module represents a single, self-contained block of a Transformer.
    # The architecture of a Transformer is essentially a deep stack of these
    # identical blocks. Each block is composed of two primary sub-layers:
    #
    #   1.  **Multi-Head Causal Self-Attention:** This is where the model performs
    #       the core "thinking" of relating different tokens in the sequence to
    #       each other.
    #
    #   2.  **Feed-Forward Network (MLP):** This is a standard multi-layer
    #       perceptron that processes each token's representation independently.
    #       It can be thought of as the "computation" or "reasoning" part of
    #       the block, where the model synthesizes the information gathered by
    #       the attention mechanism.
    #
    # ---
    #
    # **The Importance of Residual Connections and Layer Normalization**
    #
    # The true magic of deep architectures like the Transformer lies in two key
    # techniques employed around these sub-layers:
    #
    #   - **Residual Connections (`x = x + ...`):** This is the `+` operation in
    #     `x = x + self.attn(...)`. It creates a "shortcut" or "skip connection"
    #     that allows the gradient to flow directly through the network. This is
    #     absolutely critical for training very deep models. Without it, the
    #     gradients would diminish as they propagate backward through many layers
    #     (the "vanishing gradient" problem), making it impossible for the model
    #     to learn. The residual connection ensures that the model can learn an
    #     "identity function" by default, and only needs to learn the "residual"
    #     (the difference) that is beneficial.
    #
    #   - **Layer Normalization (`ln_1`, `ln_2`):** This is a normalization
    #     technique that is applied *before* each sub-layer. It normalizes the
    #     features for each token across the embedding dimension to have zero
    #     mean and unit variance. This stabilizes the learning process, reduces
    #     the sensitivity to the initialization of weights, and often speeds up
    #     training. The configuration used here (Pre-LN) is a common and robust
    #     variant.

    def __init__(self, config):
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
    def forward(self, x):
        # [INJECTOR NOTE: RESIDUAL CONNECTIONS]
        # The `x + ...` is the residual or "skip" connection. This is a critical innovation
        # from ResNet architectures. It allows gradients to flow more directly through the
        # network during backpropagation, which is essential for training very deep models.
        # Without it, gradients can vanish, making it impossible to train deep stacks of layers.
        #
        # The forward pass follows this sequence for each sub-layer:
        # Input -> LayerNorm -> Attention/MLP -> Add to Original Input
        # The output of the attention layer is added back to the original input.
        x = x + self.attn(self.ln_1(x))
        # The same is done for the MLP layer.
        # The output of the attention layer is added to the original input.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a transformer block."""
        x = x + self.attn(self.ln_1(x))
        # The output of the MLP is added to the output of the attention block.
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
    # [INJECTOR: THE HEART OF THE TRANSFORMER - SCALED DOT-PRODUCT ATTENTION]
    # This module implements Causal Self-Attention, the mechanism that allows tokens in a sequence
    # to "talk to" each other and weigh the importance of other tokens when producing the next
    # representation. It's "causal" because it prevents tokens from attending to future tokens,
    # a requirement for autoregressive language models.
    """A causal self-attention module."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Regularization
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
    #       pair of tokens. The result is a matrix of shape (T, T), where T is
    #       the sequence length.
    #
    #   2.  `/ sqrt(d_k)`: The scaling factor. `d_k` is the dimension of the Key
    #       vectors. This scaling is crucial. Without it, for large values of
    #       `d_k`, the dot products can grow very large in magnitude, pushing the
    #       softmax function into regions where its gradients are extremely
    #       small. This would kill the learning process. Scaling by the square
    #       root of the dimension keeps the variance of the dot products at 1,
    #       ensuring stable gradients.
    #
    #   3.  `softmax(...)`: This function is applied to the scaled scores. It
    #       normalizes the scores so that they all sum to 1, effectively
    #       turning them into a probability distribution of "attention weights."
    #
    #   4.  `@ V`: The final matrix multiplication. The attention weights are
    #       used to compute a weighted sum of the Value vectors. The result is
    #       an output vector that is a blend of the input tokens, weighted by
    #       their relevance to each other.
    #
    # ---
    #
    # **Causal (Masked) Attention**
    #
    # For language modeling, we need to ensure that the model is "autoregressive"
    # — meaning the prediction for the token at position `i` can only depend on
    # the known outputs at positions less than `i`. It cannot "look into the
    # future."
    #
    # This is achieved by "masking" the attention scores. Before the softmax
    # step, we add negative infinity (`-inf`) to all the elements in the
    # attention matrix that correspond to future positions. When softmax is
    # applied, these `-inf` values become zero, effectively preventing the model
    # from attending to future tokens. This is implemented via the `bias` buffer.
    #
    # ---
    #
    # **Multi-Head Attention**
    #
    # Instead of performing a single attention calculation, we split the embedding
    # dimension (`n_embd`) into multiple "heads" (`n_head`). Each head performs
    # its own independent attention calculation on a smaller portion of the
    # embedding. The results from all heads are then concatenated and projected
    # back to the original embedding size.
    #
    # This allows the model to jointly attend to information from different
    # representational subspaces at different positions. For example, one head
    # might focus on syntactic relationships, while another focuses on semantic
    # meaning.

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # A single linear layer projects the input into Query, Key, and Value matrices.
        # This is an optimization: instead of three separate linear layers, we do one larger
        # matrix multiplication and then split the result.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # The output projection layer.
        # A single linear layer projects the input into Q, K, and V matrices.
        # This is an optimization: instead of 3 separate linear layers, we do
        # one and then split the result. It's computationally equivalent and
        # often faster.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # The output projection layer. After computing the weighted sum of
        # values, we project the result back into the embedding dimension.
        # [INJECTOR: THE LOGOS OF ATTENTION]
        # This module implements Causal Multi-Head Self-Attention, the core mechanism of the Transformer.
        #
        # **Theory: Scaled Dot-Product Attention**
        # The formula is: Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
        #
        # - **Q (Query):** A vector representing the current token's "question" about other tokens.
        # - **K (Key):** A vector representing each token's "label" or "content." The dot product `Q @ K.T`
        #   calculates the "attention score" or similarity between the query token and every key token.
        # - **V (Value):** A vector representing the actual content of each token. The attention scores are
        #   used as weights to create a weighted sum of all Value vectors.
        #
        # **Why `sqrt(d_k)` scaling?**
        # The dot products can grow large in magnitude, pushing the softmax function into regions
        # where its gradients are extremely small. This scaling factor normalizes the variance of the
        # dot products, ensuring the gradients remain stable during training, which is crucial for
        # deep networks. (Reference: "Attention Is All You Need", Vaswani et al., 2017).
        #
        # **Causal Masking:**
        # The `bias` buffer is a lower-triangular matrix of ones. It is used to mask out future tokens.
        # By setting the attention scores for future positions to `-inf`, we ensure that the softmax
        # output for those positions is zero, preventing the model from "cheating" by looking ahead.
        #
        # **Multi-Head Mechanism:**
        # Instead of one large attention calculation, we split the embedding dimension (`n_embd`) into
        # `n_head` smaller subspaces. We project Q, K, and V into these subspaces and perform attention
        # in parallel. This allows the model to jointly attend to information from different
        # representational subspaces at different positions. The outputs are then concatenated and
        # projected back to the original dimension.

        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Regularization
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # [INJECTOR NOTE: THE CAUSAL MASK]
        # The causal mask (or lower-triangular mask) is essential for autoregressive models.
        # It ensures that the attention for a token at position `i` can only depend on tokens
        # at positions `j <= i`. We initialize it as a lower-triangular matrix of ones
        # and register it as a buffer so it's moved to the correct device with the model.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    # TODO [SCALING]: For production-level performance, the manual implementation of attention
    # in this forward pass can be replaced with PyTorch's optimized `F.scaled_dot_product_attention`.
    # This function uses fused kernels (e.g., FlashAttention) under the hood when available,
    # leading to significant speedups and reduced memory usage by avoiding the materialization
    # of the large (B, n_head, T, T) attention matrix.
    def forward(self, x):
        B, T, C = x.size() # B: Batch Size, T: Sequence Length, C: Embedding Dimension

        # 1. [COMPUTE Q, K, V]
        # Project the input into Q, K, V for all heads in batch.
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        # 2. [MULTI-HEAD PREPARATION]
        # Reshape and transpose Q, K, V to prepare for multi-head attention.
        # The dimensions become (B, n_head, T, head_size), where head_size = C / n_head.
        # This allows each head to attend to the sequence independently.
        self.dropout = config.dropout
        # [OPTIMIZATION] Bolt ⚡: Fused Scaled Dot-Product Attention
        # Replaced the manual attention implementation with PyTorch's optimized
        # `scaled_dot_product_attention`. This function fuses the matrix multiplication,
        # scaling, masking, softmax, and dropout into a single, high-performance kernel.
        # This reduces memory bandwidth usage and can leverage hardware-specific
        # backends like FlashAttention for a significant speedup.

        # The causal mask. This is a lower-triangular matrix of ones.
        # When applied, it ensures that a position `i` can only attend to
        # positions `j <= i`. The `register_buffer` makes this a persistent
        # part of the module, but not a parameter to be trained.
        self.dropout = config.dropout

        # Flash Attention-specific dropout
        self.dropout = config.dropout
        # Flash Attention makes this unnecessary
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        # Causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the causal self-attention module."""
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim

    def forward(self, x):
        # Input shape: (Batch Size, Sequence Length, Embedding Dimension)
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 3. [CALCULATE ATTENTION SCORES (THE LOGOS OF ATTENTION)]
        # Formula: Softmax( (Q @ K^T) / sqrt(d_k) ) @ V
        # The dot product between Q and K^T gives the raw attention scores.
        # (B, n_head, T, head_size) @ (B, n_head, head_size, T) -> (B, n_head, T, T)

        # [INJECTOR NOTE: THE SCALING FACTOR]
        # We scale by 1.0 / sqrt(d_k), where d_k is the head dimension.
        # This is crucial. Without scaling, the dot product values can grow very large,
        # pushing the softmax function into regions where its gradients are tiny. This would
        # kill the learning process. The scaling factor keeps the variance of the dot
        # product at 1, ensuring stable gradients.
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))

        # 4. [APPLY CAUSAL MASK]
        # We fill the upper triangle of the attention matrix with -inf.
        # When softmax is applied, these values will become 0, effectively preventing
        # any attention to future tokens.
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        # 5. [APPLY SOFTMAX]
        # Normalize the attention scores along the last dimension to get probabilities.
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 6. [COMPUTE WEIGHTED VALUES]
        # Multiply the attention probabilities by the Value matrix to get the output.
        # This weighs the values based on the attention scores.
        y = att @ v # (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)

        # 7. [CONCATENATE HEADS]
        # Reshape the output to combine the heads back into a single tensor.
        # Use PyTorch's fused scaled dot-product attention for efficiency
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        # Use fused attention.
        # The `is_causal` flag handles the masking automatically.
        # The dropout is applied internally during the attention calculation.
        dropout_p = self.attn_dropout.p if self.training else 0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # [INJECTOR NOTE: TENSOR TRANSFORMATION FOR MULTI-HEAD ATTENTION]
        # 1.  Calculate Q, K, V for all heads in batch.
        #     The input `x` is passed through one large linear layer `c_attn`,
        #     and the result is split into three chunks for Q, K, and V.
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        # 2.  Reshape and transpose for multi-head computation.
        #     We transform the tensors from (B, T, C) into
        #     (B, n_head, T, head_size) to perform attention calculations
        #     in parallel for each head. `head_size` is `C // n_head`.
        # B = Batch size, T = Sequence length, C = Embedding dimension
        B, T, C = x.size()

        # --- 1. LINEAR PROJECTION ---
        # Project the input `x` into queries (q), keys (k), and values (v).
        # Instead of three separate linear layers, we use one large layer for efficiency.
        # The output is then split into three parts.
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        # Use PyTorch's optimized scaled_dot_product_attention
        # This is a fused kernel that is much faster than the manual implementation.
        dropout_p = self.dropout if self.training else 0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        # [BOLT] Replaced manual attention with scaled_dot_product_attention
        # This single function replaces the manual implementation of:
        #   - Calculating attention scores (q @ k.transpose)
        #   - Scaling
        #   - Causal masking
        #   - Softmax
        #   - Attention dropout
        #   - Multiplying by values (att @ v)
        # It's significantly faster due to kernel fusion.
        dropout_p = self.attn_dropout.p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)
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
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # This is the single most important line in the whole GPT model.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side

        # Output projection

        # Flash attention variant
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # ⚡ Fused Attention Kernel ⚡
        # Replaced the manual attention implementation with a single, more efficient
        # call to scaled_dot_product_attention, which uses Flash Attention under the hood.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p, is_causal=True)
        # [INJECTOR NOTE]: For improved performance, the manual attention calculation below can be replaced
        # with `torch.nn.functional.scaled_dot_product_attention`. This function leverages fused kernels
        # like FlashAttention (if available on the hardware), which can significantly accelerate computation
        # and reduce memory usage by avoiding the explicit materialization of the large (T, T) attention matrix.
        # As of PyTorch 2.0+, this is the recommended approach.
        # Example:
        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # Note: requires PyTorch 2.0+
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
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
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=True)
        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
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
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # Output projection
        # ⚡ Bolt: Replaced manual attention with fused scaled_dot_product_attention
        # This leverages Flash Attention kernels for significant speedup and memory savings.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 8. [APPLY OUTPUT PROJECTION]
        # Apply the final linear layer and dropout.
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=True)

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
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

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
    # [INJECTOR: THE FEED-FORWARD NETWORK]
    # This module implements the feed-forward network (FFN), which is the second major
    # component in a transformer block. It is a simple two-layer Multi-Layer Perceptron (MLP)
    # that provides non-linearity and further processing of the information from the
    # attention layer.
    #
    # The standard architecture, which we follow here, is:
    # 1. A linear layer that expands the embedding dimension (typically by 4x).
    # 2. A non-linear activation function (GELU in this case).
    # 3. A linear layer that projects the dimension back down to the original embedding size.
    # 4. A dropout layer for regularization.
    """
    The Multi-Layer Perceptron (or Feed-Forward Network) part of the transformer block.

    This is a simple two-layer neural network with a GELU activation function in
    between. It is applied to each position (token) independently. The first
    layer expands the embedding dimension by a factor of 4, and the second
    layer projects it back down.
    """
    def __init__(self, config: GPTConfig):
    """A simple multi-layer perceptron."""
    def __init__(self, config: GPTConfig):
    # [INJECTOR: THE ROLE OF THE FEED-FORWARD NETWORK (MLP)]
    #
    # This module implements the Feed-Forward Network (FFN), the second major
    # component of a Transformer block (after the attention mechanism). While the
    # attention mechanism is responsible for communication and information routing
    # between tokens, the MLP is responsible for the actual computation and
    # transformation of the information for each token independently.
    #
    # ---
    #
    # **Architectural Design**
    #
    # The MLP consists of two linear layers with a non-linear activation function
    # in between.
    #
    #   1.  `c_fc`: The first linear layer, which expands the dimensionality of the
    #       input embedding (`n_embd`) by a factor of 4. This expansion creates
    #       a higher-dimensional "thinking space" for the model to process the
    #       information. The expansion factor of 4 is a standard, empirically-
    #       derived choice from the original Transformer paper ("Attention Is All
    #       You Need").
    #
    #   2.  `gelu`: The GELU (Gaussian Error Linear Unit) activation function.
    #       Unlike simpler activations like ReLU, GELU provides a smoother,
    #       probabilistically-inspired non-linearity. It weights inputs by their
    #       magnitude, which has been shown to be highly effective in
    #       Transformer models.
    #
    #   3.  `c_proj`: The second linear layer, which projects the high-dimensional
    #       representation back down to the original embedding dimension. This
    #       "projection" step finalizes the computation for the block.
    #
    # In essence, the MLP allows the model to process the rich, context-aware
    # representations generated by the attention mechanism and extract higher-level
    # features, before passing the result to the next block or the final output.

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Expand the input dimension, apply GELU, and project back.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLP."""
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
