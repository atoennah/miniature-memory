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
    # [INJECTOR: THE BUILDING BLOCK OF THE TRANSFORMER]
    # This module represents a single block of a transformer. It consists of two main sub-layers:
    # 1. A Causal Self-Attention layer.
    # 2. A Feed-Forward Network (MLP).
    #
    # Each of these sub-layers has a residual connection ("skip connection") around it, followed
    # by a Layer Normalization. This is a "Pre-LayerNorm" architecture, where the normalization
    # is applied *before* the main operation (attention or MLP).

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # [INJECTOR NOTE: RESIDUAL CONNECTIONS]
        # The `x + ...` is the residual or "skip" connection. This is a critical innovation
        # from ResNet architectures. It allows gradients to flow more directly through the
        # network during backpropagation, which is essential for training very deep models.
        # Without it, gradients can vanish, making it impossible to train deep stacks of layers.
        #
        # The forward pass follows this sequence for each sub-layer:
        # Input -> LayerNorm -> Attention/MLP -> Add to Original Input
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CausalSelfAttention(nn.Module):
    # [INJECTOR: THE HEART OF THE TRANSFORMER - SCALED DOT-PRODUCT ATTENTION]
    # This module implements Causal Self-Attention, the mechanism that allows tokens in a sequence
    # to "talk to" each other and weigh the importance of other tokens when producing the next
    # representation. It's "causal" because it prevents tokens from attending to future tokens,
    # a requirement for autoregressive language models.

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # A single linear layer projects the input into Query, Key, and Value matrices.
        # This is an optimization: instead of three separate linear layers, we do one larger
        # matrix multiplication and then split the result.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # The output projection layer.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
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
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 8. [APPLY OUTPUT PROJECTION]
        # Apply the final linear layer and dropout.
        y = self.resid_dropout(self.c_proj(y))
        return y

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

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Expand the input dimension, apply GELU, and project back.
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
