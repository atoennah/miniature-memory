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

class FeedForward(nn.Module):
    """A simple feed-forward network module."""
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

class CausalSelfAttention(nn.Module):
    # [INJECTOR: THE LOGOS OF CAUSAL SELF-ATTENTION]
    #
    # This module is the cornerstone of the Transformer. It allows the model to weigh the
    # importance of different tokens in the input sequence when producing a representation
    # for a given token. It is "causal" because it is masked to prevent attending to
    # future tokens, which is essential for autoregressive language generation.
    #
    # The core mechanism is Scaled Dot-Product Attention. The formula is:
    #   Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    #
    # Breakdown of the components:
    # 1. Q (Query): A projection of the current token's embedding. It's the "question"
    #    this token is asking about other tokens.
    # 2. K (Key): A projection of all tokens' embeddings in the sequence. It's the "label"
    #    or "identifier" for each token. The dot product `Q @ K.T` measures the similarity
    #    or "resonance" between the query token and all key tokens.
    # 3. V (Value): A projection of all tokens' embeddings. It contains the actual information
    #    that should be aggregated.
    #
    # The Scaling Factor `sqrt(d_k)`:
    # `d_k` is the dimension of the key vectors. We divide by its square root to stabilize
    # the gradients. For large `d_k`, the dot products `Q @ K.T` can grow very large in
    # magnitude, pushing the softmax function into regions where its gradients are
    # vanishingly small. This scaling prevents that saturation, ensuring stable training.
    #
    # Multi-Head Attention:
    # Instead of one large attention calculation, we split the embedding dimension (`n_embd`)
    # into `n_head` smaller "heads." Each head performs its own attention calculation in
    # parallel on a subspace of the embeddings. This allows the model to jointly attend to
    # information from different representational subspaces at different positions. The outputs
    # of all heads are then concatenated and projected back to the original dimension.
    #
    # Causal Mask (`is_causal=True`):
    # In the `F.scaled_dot_product_attention` call, this argument automatically applies a
    # mask to the attention scores. The mask is an upper-triangular matrix of negative
    # infinity, which, after the softmax, zeros out the scores for all tokens that appear
    # *after* the current token in the sequence. This ensures the model is autoregressive.

    """A causal self-attention module with multi-head support."""
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
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape and transpose for multi-head attention
        # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal self-attention using PyTorch's fused kernel
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        # Re-assemble all head outputs side by side
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """A single Transformer block."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a Transformer block."""
        # [INJECTOR: RESIDUALS & PRE-LAYER NORMALIZATION]
        #
        # This is the standard "Pre-LN" Transformer block architecture, popularized by GPT-2.
        # Each sub-layer (attention and MLP) has its own Layer Normalization applied *before*
        # the transformation, and is wrapped in a residual connection.
        #
        # 1. Residual Connection (x + ...): This is the "shortcut" or "skip-connection"
        #    that allows gradients to flow directly through the network. Without this,
        #    training very deep transformers (>12 layers) becomes unstable due to the
        #    vanishing gradient problem. It allows the model to learn modifications to the
        #    identity function, rather than learning the entire transformation from scratch.
        #
        # 2. Pre-LN (self.ln_1(x)): Normalizing the input *before* the attention/MLP helps
        #    stabilize the training dynamics, leading to smoother convergence and often
        #    allowing for higher learning rates. The output of the sub-layer is then added
        #    back to the *original*, un-normalized input `x`.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """A GPT-style transformer model."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # [INJECTOR: WEIGHT TYING]
        #
        # Weight tying is a technique where the token embedding matrix (`wte`) and the final
        # output projection matrix (`lm_head`) share the same weights.
        #
        # Conceptual Rationale:
        # The embedding layer maps a token ID to a vector representation. The output layer
        # maps from a vector representation back to a distribution over token IDs (logits).
        # Intuitively, these two operations are doing the inverse of each other. Tying the
        # weights forces them to learn the same representation, which can:
        #   1. Reduce the total number of parameters in the model, which is especially
        #      significant for large vocabularies.
        #   2. Improve model quality by regularizing it and preventing the input and output
        #      representations from drifting apart.
        # This is a standard practice in modern language models.
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # [INJECTOR: GPT-2 RESIDUAL PROJECTION SCALING]
        #
        # This is a specific initialization scheme from the GPT-2 paper. We apply a
        # scaled normal initialization to the weights of the residual projection layers
        # (`c_proj` in both attention and MLP).
        #
        # The standard deviation is scaled by `1 / sqrt(N)`, where N is the number of
        # residual layers (`2 * n_layer` because there's one in attention and one in MLP).
        #
        # Rationale:
        # At initialization, the residual connections should be close to an identity function.
        # By scaling down the weights of the projection layers that are on the residual path,
        # we ensure that their output is small at the beginning of training. This helps
        # stabilize the initial training dynamics and prevents the model's outputs from
        # exploding in magnitude.
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
