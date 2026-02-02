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
#
# [INJECTOR NOTE]: For massive scale (10B+ parameters), we would transition from
# this vanilla implementation to Megatron-LM style tensor parallelism or
# DeepSpeed ZeRO-3 to handle memory sharding across multiple GPUs.

"""
A minimal, from-scratch GPT model implementation.
Based on Andrej Karpathy's NanoGPT: https://github.com/karpathy/nanogpt
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple, List, Dict

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
    # and relationships.
    # Reference: Vaswani et al. (2017) - https://arxiv.org/abs/1706.03762
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
    #     it asks of other tokens.
    # 2.  K (Key): A projection of every token's embedding. It's the "label" or "identifier".
    # 3.  V (Value): Another projection of every token's embedding. It contains the actual
    #     information.
    #
    # / sqrt(d_k): This is a critical scaling factor. Without it, the dot products can
    # become very large, pushing the softmax into regions with extremely small gradients.
    #
    # Causal Masking (`is_causal=True`): Ensures the model is autoregressive by
    # preventing it from "cheating" and looking at future tokens.
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

        # [INJECTOR: DEMYSTIFYING MULTI-HEAD TENSOR MANIPULATION]
        #
        # Multi-head attention runs the mechanism in parallel several times, allowing
        # the model to attend to information from different subspaces.
        # Transformations: (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        #
        # [INJECTOR NOTE]: For future optimization, consider implementing
        # Grouped-Query Attention (GQA) or Multi-Query Attention (MQA) to
        # reduce the KV-cache memory footprint, as seen in Llama-2 and Falcon.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # [INJECTOR: THE LOGOS OF KV-CACHING]
        #
        # Autoregressive generation is naturally sequential. Key-Value (KV) Caching
        # recognizes that for past tokens, the Key (K) and Value (V) projections
        # remain constant.
        # Logic: Append current K/V to cache, perform attention against full history.
        # This transforms per-token cost from O(T) to O(1).
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        new_kv_cache = (k, v)

        # [INJECTOR: THE LOGOS OF THE KERNEL]
        #
        # We utilize `F.scaled_dot_product_attention` (SDPA), which is a "fused kernel".
        # It implements 'FlashAttention' (Dao et al., 2022) to avoid materializing
        # the O(T^2) attention matrix in high-bandwidth memory (HBM).
        # By keeping intermediate results in SRAM, we achieve O(T) memory complexity
        # and massive speedups.
        # Reference: https://arxiv.org/abs/2205.14135

        # [INJECTOR NOTE]: For generation with extremely long context, look into
        # FlashAttention-2 or FlashAttention-3 for even higher throughput.

        is_causal = (x.size(1) > 1 and kv_cache is None)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=is_causal)

        # Re-assemble all head outputs side by side
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kv_cache

class Block(nn.Module):
    # [INJECTOR: THE ARCHITECTURE OF A TRANSFORMER BLOCK]
    #
    # A Transformer is essentially a deep stack of these "Block" modules.
    # Two key innovations make training deep stacks possible:
    #
    # 1. Residual Connections (The `+`): Also known as "skip connections".
    #    Instead of forcing the output of a sub-layer to represent the *entire*
    #    desired output, we only ask it to learn the *residual* or the *change*.
    #    This creates a direct path for gradients to flow during backprop,
    #    solving the vanishing gradient problem.
    #
    # 2. Pre-Layer Normalization: Stabilizes inputs to each sub-layer.
    #    We adopt "Pre-LN" architecture (normalizing before the main operation),
    #    which is more stable for deep models than the original "Post-LN".
    """A single Transformer block."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for a Transformer block."""
        attn_out, new_kv_cache = self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + attn_out
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
            # [INJECTOR: THE NECESSITY OF POSITIONAL EMBEDDINGS]
            #
            # Attention is permutation-invariant. Positional embeddings inject
            # information about *where* a token is. We use learnable absolute
            # positional embeddings here (GPT-2 style).
            #
            # [INJECTOR NOTE]: In modern architectures (e.g., Llama, Mistral),
            # we would use Rotary Positional Embeddings (RoPE) to allow
            # better extrapolation to longer sequence lengths.
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # [INJECTOR: THE EFFICIENCY OF WEIGHT TYING]
        #
        # Sharing weights between the embedding (wte) and output head (lm_head)
        # dramatically reduces parameter count (~38M saved for vocab=50k, d=768)
        # and typically improves performance by linking input/output spaces.
        # Reference: Press & Wolf (2016) - https://arxiv.org/abs/1608.05859
        self.transformer.wte.weight = self.lm_head.weight

        # Pre-compute positional indices and register as a buffer
        pos = torch.arange(0, config.block_size, dtype=torch.long).unsqueeze(0)
        self.register_buffer('pos', pos, persistent=False)

        # init all weights
        self.apply(self._init_weights)

        # [INJECTOR: GPT-2 STYLE WEIGHT INITIALIZATION]
        # Scaling the residual projection layers by sqrt(2 * layers) ensures
        # the residual path doesn't explode at initialization.
        # Reference: GPT-2 paper - https://d4mucfpotywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
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

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for the GPT model."""
        b, t = idx.size()

        if kv_caches is not None:
            # [INJECTOR NOTE]: Ensure past_length + t <= block_size to avoid
            # OOB errors in positional embedding lookup.
            past_length = kv_caches[0][0].size(2)
            pos = self.pos[:, past_length : past_length + t]
        else:
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            pos = self.pos[:, :t]

        # Token and position embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        new_kv_caches = []
        for i, block in enumerate(self.transformer.h):
            block_kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, kv_cache=block_kv)
            new_kv_caches.append(new_kv)

        # Final layer norm and language model head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_p: float = 0.9) -> torch.Tensor:
        """
        Autoregressively generates a sequence of tokens using top-p (nucleus) sampling
        and an efficient KV-cache.
        """
        self.eval()

        # [INJECTOR: THE LOGOS OF NUCLEUS SAMPLING]
        #
        # Nucleus Sampling (Top-p) finds the "Goldilocks" zone by dynamically
        # selecting the smallest set of tokens whose cumulative probability
        # exceeds 'p'. This adapts to the model's confidence.
        # Reference: Holtzman et al. (2019) - https://arxiv.org/abs/1904.09751

        # [INJECTOR NOTE]: For high-throughput production, consider
        # Speculative Decoding to accelerate generation by using a small
        # draft model to predict multiple tokens at once.

        kv_caches = None

        for i in range(max_new_tokens):
            idx_cond = idx[:, -1:] if kv_caches is not None else idx

            current_pos = idx.size(1)
            if current_pos >= self.config.block_size:
                kv_caches = None
                idx_cond = idx[:, -self.config.block_size:]

            logits, _, kv_caches = self(idx_cond, kv_caches=kv_caches)
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
