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
    # [INJECTOR: THE ROLE OF THE FEED-FORWARD NETWORK]
    #
    # While the self-attention mechanism is responsible for communication between tokens,
    # the Feed-Forward Network (FFN) is responsible for the "computation" or "thinking"
    # on a per-token basis. It is a simple two-layer Multi-Layer Perceptron (MLP) that
    # is applied independently to each token's representation.
    #
    # The structure is:
    # 1.  A linear layer that expands the embedding dimension (`n_embd`) by a factor of 4.
    # 2.  A non-linear activation function (GELU in this case).
    # 3.  A linear layer that projects the result back down to the original embedding dimension.
    #
    # Why the 4x expansion?
    # This is a convention established in the original "Attention Is All You Need" paper.
    # The expansion creates a "bottleneck" architecture where the model has a much larger
    # internal space to process and transform features for each token. This added
    # computational capacity is crucial for the model's ability to learn complex patterns
    # and relationships. While other expansion factors can be used, 4x has been found to be
    # a robust and effective choice for a wide range of tasks.
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
    # At the heart of the Transformer is the ability to weigh the importance of different
    # tokens in a sequence when producing a representation for a specific token. This
    # mechanism is called Self-Attention.
    #
    # The formula for Scaled Dot-Product Attention is:
    #   Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    #
    # Breakdown of the components:
    # 1.  Q (Query): A projection of the current token's embedding. It's the "question"
    #     it asks of other tokens: "What information do you have that is relevant to me?"
    # 2.  K (Key): A projection of every token's embedding in the sequence. It's the "label"
    #     or "identifier" for each token, advertising its contents.
    # 3.  V (Value): Another projection of every token's embedding. It contains the actual
    #     information that a token holds.
    #
    # The process:
    # -   (Q @ K.T): The dot product between the Query of one token and the Keys of all
    #     other tokens measures their "similarity" or "resonance". A high value means
    #     the key's token is highly relevant to the query's token.
    # -   / sqrt(d_k): This is a critical scaling factor. `d_k` is the dimension of the
    #     key vectors. Without this scaling, the dot products can become very large,
    #     pushing the softmax function into regions where its gradients are tiny. This
    #     would kill the learning process. This scaling keeps the variance of the dot
    #     products at 1, stabilizing training.
    # -   softmax(...): This turns the raw similarity scores into a probability
    #     distribution (a set of weights that sum to 1). A token with a high score
    #     gets a high weight, and vice-versa.
    # -   ... @ V: The final output is a weighted sum of the Value vectors. Tokens
    #     deemed more relevant (with higher softmax weights) contribute more of their
    #     "value" to the final representation of the current token.
    #
    # Causal Masking (`is_causal=True`):
    # For a language model that predicts the next token, we must prevent it from "cheating"
    # by looking at future tokens. The causal mask is an operation that adds -infinity
    # to the attention scores for all tokens that come *after* the current token. When
    # the softmax is applied, these -infinity scores become zero, effectively masking out
    # any information from the future.
    """A causal self-attention module with multi-head support."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # Regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for the causal self-attention module."""
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if kv_cache is not None:
            # Concatenate with past key and value tensors
            past_k, past_v = kv_cache
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        # [INJECTOR: KV CACHE DETACHMENT]
        # We detach the key and value tensors from the computation graph before storing
        # them in the cache. This is a crucial optimization for inference. It prevents
        # PyTorch from tracking the history of the entire generated sequence, which would
        # otherwise lead to a linear increase in memory usage with every new token.
        # By detaching, we treat the past as a fixed constant, ensuring that the memory
        # footprint remains small and constant during autoregressive generation.
        current_kv_cache = (k.detach(), v.detach())

        # [INJECTOR: DYNAMIC CAUSAL MASKING]
        # The `is_causal` flag in `scaled_dot_product_attention` is highly efficient, but
        # it should only be used during the initial "prefill" step where the prompt has
        # a sequence length (T) greater than 1. For subsequent generation steps, T is 1,
        # and a causal mask is unnecessary (a single token attending to itself and the
        # past doesn't need masking). Disabling it for T=1 provides a minor speedup.
        is_causal = T > 1

        # Causal self-attention using PyTorch's fused kernel
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=is_causal)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, current_kv_cache

class Block(nn.Module):
    """A single Transformer block."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for a Transformer block."""
        attn_output, new_kv_cache = self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache

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

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, kv_caches: Optional[list] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """Forward pass for the GPT model."""
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # [INJECTOR: KV CACHE POSITION EMBEDDING ADJUSTMENT]
        # When using a KV cache, the position embeddings need to be handled carefully.
        # During the initial "prefill" step, `t` is the full prompt length.
        # During subsequent "generation" steps, `t` is 1.
        # The position embedding must correspond to the token's actual position in the
        # full sequence, not its position in the current input tensor.
        # We determine the starting position from the size of the cache, if it exists.
        past_length = 0
        if kv_caches is not None and kv_caches[0] is not None:
             past_length = kv_caches[0][0].size(2) # Get T from (B, nh, T, hs)
        pos = self.pos[:, past_length : past_length + t]

        # Token and position embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        new_kv_caches = []
        for i, block in enumerate(self.transformer.h):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_layer_cache = block(x, kv_cache=layer_cache)
            new_kv_caches.append(new_layer_cache)

        # Final layer norm and language model head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, new_kv_caches

    def _sample_next_token(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        """Helper function to sample the next token using top-p sampling."""
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

        # Top-p (nucleus) sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[:, indices_to_remove] = 0
        probs = probs / torch.sum(probs, dim=-1, keepdim=True)

        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_p: float = 0.9) -> torch.Tensor:
        """
        Autoregressively generates a sequence of tokens using top-p (nucleus) sampling.
        This implementation uses a KV cache to dramatically speed up generation.

        The process is divided into two phases:
        1.  Prefill Phase: The initial prompt (`idx`) is processed in a single forward
            pass. This populates the KV cache with the keys and values for all tokens
            in the prompt. This is the most computationally expensive part.
        2.  Generation Phase: For each new token, we perform a forward pass on only
            the *single* most recently generated token. We pass in the cache from the
            previous step, which contains the keys and values for all preceding tokens.
            This avoids re-computing the attention for the entire sequence at every
            step, making generation much faster.
        """
        self.eval()

        # Initialize the KV cache for all layers
        kv_caches = [None] * self.config.n_layer

        # Prefill phase: process the initial prompt
        logits, _, kv_caches = self(idx, kv_caches=kv_caches)
        idx_next = self._sample_next_token(logits[:, -1, :], temperature, top_p)
        idx = torch.cat((idx, idx_next), dim=1)

        # Generation phase: generate subsequent tokens one by one
        for _ in range(max_new_tokens - 1): # `-1` because we already generated one token
            logits, _, kv_caches = self(idx_next, kv_caches=kv_caches)
            idx_next = self._sample_next_token(logits[:, -1, :], temperature, top_p)
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx
