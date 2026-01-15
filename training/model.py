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
    """
    A causal self-attention module with multi-head support.

    [INJECTOR: THE LOGOS OF SELF-ATTENTION]

    This module is the heart of the Transformer. It allows the model to weigh the
    importance of different tokens in the input sequence when producing a representation
    for each token.

    The core mechanism is Scaled Dot-Product Attention. The formula is:
        Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V

    Where:
    - Q (Query): A projection of the input representing the current token's "question"
                 about other tokens.
    - K (Key): A projection of the input representing the other tokens' "answers"
               or attributes.
    - V (Value): A projection of the input representing the actual content of the
                 other tokens.

    The dot product `Q @ K.T` computes a similarity score between each query and all keys.
    The scaling factor `1 / sqrt(d_k)` (where d_k is the head dimension) is crucial.
    Without it, for large values of d_k, the dot products can grow very large, pushing
    the softmax function into regions with extremely small gradients, which makes
    training unstable.

    Multi-Head Attention:
    Instead of performing a single attention calculation, we project Q, K, and V multiple
    times with different, learned linear projections (n_head times). This allows the
    model to jointly attend to information from different representation subspaces at
    different positions. It's like having multiple experts, each focusing on a different
    aspect of the input.

    Causal (Autoregressive) Masking:
    In a language model, the prediction for token `i` can only depend on the known outputs
    at positions less than `i`. This is achieved by masking out (setting to -infinity)
    all values in the attention scores that correspond to future positions. PyTorch's
    `F.scaled_dot_product_attention` handles this internally with the `is_causal=True` flag.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, query, value projections for all heads, but in a batch.
        # A single linear layer is used for efficiency.
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

        # [INJECTOR NOTE: BATCHED PROJECTION]
        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim.
        # This is a single matrix multiplication that projects the input `x` into Q, K, and V
        # for all heads simultaneously, which is highly efficient.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # [INJECTOR NOTE: TENSOR RESHAPING FOR MULTI-HEAD]
        # Reshape and transpose the Q, K, V tensors to prepare them for multi-head attention.
        # The dimensions are rearranged from (Batch, Time, Channels) to (Batch, Num_Heads, Time, Head_Size)
        # This allows the attention mechanism to process all heads in parallel.
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)

        # [INJECTOR NOTE: FUSED ATTENTION KERNEL]
        # Causal self-attention using PyTorch's highly optimized fused kernel.
        # This one function handles the dot product, scaling, masking, softmax, and value multiplication.
        # `is_causal=True` ensures that attention is only paid to previous tokens, making the model autoregressive.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        # [INJECTOR NOTE: REASSEMBLING THE HEADS]
        # Re-assemble all head outputs side by side. The transpose and view operations reverse the
        # reshaping done earlier, concatenating the outputs of all heads to form the final result.
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

        # Weight tying: https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
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

        # [INJECTOR: THE CONCEPT OF POSITION]
        # The Transformer architecture is permutation-invariant, meaning it has no inherent
        # sense of token order. To provide the model with sequential information, we
        # introduce positional embeddings.
        #
        # `wte` (Word Token Embeddings): Maps each token index to a dense vector representation.
        # `wpe` (Word Position Embeddings): Maps each integer position (0, 1, ..., t) to a
        #                                  dense vector.
        #
        # These two embeddings are summed element-wise to create a final representation
        # that encodes both the token's identity and its position in the sequence.
        tok_emb = self.transformer.wte(idx) # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # (1, t, n_embd)
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
