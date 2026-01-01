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
from dataclasses import dataclass

@dataclass
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

        # A single linear layer projects the input into Q, K, and V for all heads at once.
        # This is more efficient than creating separate linear layers for each head.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # A final projection layer to combine the outputs of all heads.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)


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

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # First sub-layer: Self-Attention, preceded by LayerNorm, followed by a residual connection.
        x = x + self.sa(self.ln1(x))
        # Second sub-layer: Feed-Forward, preceded by LayerNorm, followed by a residual connection.
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
