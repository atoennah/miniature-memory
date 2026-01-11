# [INJECTOR: THE PHILOSOPHY OF A FROM-SCRATCH GPT]
#
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

class GPTConfig:
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
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

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
