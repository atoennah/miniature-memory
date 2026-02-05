# [INJECTOR: THE PHILOSOPHY OF A FROM-SCRATCH GPT]
#
# This file is the pedagogical core of a from-scratch GPT implementation. Its purpose
# is not just to be functional, but to be exceptionally clear and educational. We build
# every core component of the Transformer architecture (as described in "Attention Is All
# You Need") from basic PyTorch primitives.
#
# [SCIENTIFIC PROOF: THE LOGOS OF THE TRANSFORMER]
# The Transformer architecture solved the "Sequential Bottleneck" of RNNs. In an RNN,
# the hidden state h_t depends on h_{t-1}, forcing O(N) sequential operations.
# The Transformer uses Self-Attention, allowing O(1) path length between any two tokens,
# which enables massive parallelism and solves the Vanishing Gradient problem for
# long-range dependencies.
#
# The hierarchy of the model is as follows:
# 1.  GPT (The full model)
#       - Manages token and positional embeddings.
#       - Contains a stack of Transformer Blocks.
#       - Ends with a layer normalization and a linear layer to produce logits.
#
# 2.  Block (A single Transformer layer)
#       - Encapsulates one Multi-Head Causal Self-Attention module and one MLP.
#       - Employs residual connections ("shortcuts") around both sub-modules.
#
# 3.  CausalSelfAttention (The heart of the Transformer)
#       - Implements Multi-Head Scaled Dot-Product Attention.
#       - Now includes a stateful KV-Cache for O(N) inference speed.
#
# 4.  MLP (Feed-Forward Network)
#       - Responsible for per-token non-linear feature transformation.

"""
A minimal, from-scratch GPT model implementation.
Based on Andrej Karpathy's NanoGPT: https://github.com/karpathy/nanogpt
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple, List

class GPTConfig:
    """Configuration for the GPT model."""
    def __init__(self,
                 vocab_size: int,
                 block_size: int = 1024,
                 n_layer: int = 12,
                 n_head: int = 12,
                 n_embd: int = 768,
                 dropout: float = 0.1):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

class FeedForward(nn.Module):
    # [INJECTOR: THE ROLE OF THE FEED-FORWARD NETWORK]
    #
    # While the self-attention mechanism is responsible for communication between tokens,
    # the Feed-Forward Network (FFN) is responsible for the "computation" or "thinking"
    # on a per-token basis.
    #
    # Why the 4x expansion?
    # Formula: d_ff = 4 * d_model. This is the "inner dimension" of the MLP.
    # The expansion creates a high-dimensional space where features can be separated
    # and transformed before being compressed back.
    """A simple feed-forward network module."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CausalSelfAttention(nn.Module):
    # [INJECTOR: THE LOGOS OF SELF-ATTENTION]
    #
    # Formula: Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    #
    # Why sqrt(d_k)?
    # [MATHEMATICAL PROOF]: Assume components of Q and K are independent random
    # variables with mean 0 and variance 1. Their dot product Q @ K has mean 0
    # and variance d_k. To maintain a variance of 1 (which keeps softmax gradients
    # stable), we scale by 1/sqrt(d_k).
    """A causal self-attention module with multi-head support."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional KV-cache support.

        Args:
            x: Input tensor (B, T, C)
            past_key_value: Tuple of (K, V) from previous steps (B, nh, T_prev, hs)

        Returns:
            y: Output tensor (B, T, C)
            present_key_value: Tuple of updated (K, V) (B, nh, T_curr, hs)
        """
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Multi-head split
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # [INJECTOR NOTE: KV-CACHE MECHANICS]
        # In autoregressive generation, we only care about the last token's output.
        # Instead of recomputing K and V for all previous tokens, we store them.
        # This reduces inference complexity from O(N^2) to O(N).
        if past_key_value is not None:
            prev_k, prev_v = past_key_value
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        present_key_value = (k, v)

        # Attention calculation
        # If we have a KV cache and T=1, we don't need causal masking.
        # PyTorch's SDPA handles efficient kernels like FlashAttention-2.
        is_causal = (past_key_value is None) # Only causal mask if processing full sequence
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=is_causal)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, present_key_value

class Block(nn.Module):
    """A single Transformer block."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, present_key_value = self.attn(self.ln_1(x), past_key_value=past_key_value)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_key_value

class GPT(nn.Module):
    """A GPT-style transformer model."""
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
        self.transformer.wte.weight = self.lm_head.weight

        pos = torch.arange(0, config.block_size, dtype=torch.long).unsqueeze(0)
        self.register_buffer('pos', pos, persistent=False)

        self.apply(self._init_weights)
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

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        b, t = idx.size()

        if past_key_values is not None:
            t_start = past_key_values[0][0].size(2)
            # Ensure we don't exceed block_size in positional lookup
            t_curr = min(t_start + t, self.config.block_size)
            pos = self.pos[:, t_start : t_curr]
            # If t was > 1 but we hit the cap, we might need to truncate tok_emb too
            # but in generate, t is always 1 when using cache.
        else:
            pos = self.pos[:, :t]

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        present_key_values = []
        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, past_key_value=past_kv)
            present_key_values.append(present_kv)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, present_key_values

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_p: float = 0.9, use_cache: bool = True) -> torch.Tensor:
        """
        Autoregressively generates a sequence of tokens.
        """
        self.eval()
        past_key_values = None

        for _ in range(max_new_tokens):
            if use_cache:
                if past_key_values is not None:
                    # Current total length = past_length + 1
                    past_length = past_key_values[0][0].size(2)
                    if past_length >= self.config.block_size:
                        # Truncate cache to allow for one new token (sliding window)
                        past_key_values = [ (k[:, :, 1:, :], v[:, :, 1:, :]) for k, v in past_key_values ]
                    idx_cond = idx[:, -1:]
                else:
                    # Initial prompt
                    idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

                logits, _, past_key_values = self(idx_cond, past_key_values=past_key_values)
            else:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                logits, _, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            # Top-p (nucleus) sampling
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[:, indices_to_remove] = 0
            probs = probs / torch.sum(probs, dim=-1, keepdim=True)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx
