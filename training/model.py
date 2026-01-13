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
    # The Feed-Forward Network (FFN), or MLP, is the second key component of a
    # Transformer block (the first being the attention mechanism). While attention
    # handles the mixing of information across the sequence length, the FFN is
    # responsible for processing and transforming the features at each token position
    # independently.
    #
    # The architecture follows a simple "expand-and-contract" pattern:
    # 1.  An initial Linear layer projects the input embedding (`n_embd`) to a much
    #     larger inner dimension (`4 * n_embd`). This expansion allows the model to
    #     learn a richer set of features and relationships. The factor of 4 is a
    #     convention established by the original "Attention Is All You Need" paper.
    #
    # 2.  A non-linear activation function (GELU) is applied. The Gaussian Error
    #     Linear Unit (GELU) is a high-performing activation that weights inputs by
    #     their magnitude, unlike ReLU which gates them sharply at zero. This smooth
    #     probabilistic gating often leads to better performance.
    #
    # 3.  A final Linear layer projects the expanded representation back down to the
    #     original embedding dimension (`n_embd`), allowing the output to be added
    #     to the residual path.
    #
    # In essence, the FFN acts as a content-based feature extractor, giving the model
    # the capacity to "think" about the information it has gathered via self-attention.
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
    # This module implements Multi-Head Causal Scaled Dot-Product Attention, the
    # cornerstone of the Transformer architecture.
    #
    # The core operation is the Scaled Dot-Product Attention formula:
    #   Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    #
    # Breakdown of the components:
    # 1.  Q (Query), K (Key), V (Value): These are projections of the input tensor `x`.
    #     Each token in the sequence creates a query, a key, and a value.
    #       - The Query represents what a token is "looking for."
    #       - The Key represents what a token "contains."
    #       - The Value represents what a token "offers" once attention is paid.
    #
    # 2.  Dot Product (Q @ K.T): The dot product between a query and all keys measures
    #     their similarity or "affinity." A high dot product means the query and key
    #     are closely related. This produces a raw "attention score" matrix.
    #
    # 3.  Scaling (/ sqrt(d_k)): The `d_k` is the dimension of the key vectors. We scale
    #     the attention scores to prevent their magnitudes from becoming too large.
    #     Large values would push the softmax function into regions where its gradients
    #     are near zero, effectively stalling the learning process. This is a critical
    #     stabilization technique.
    #
    # 4.  Causal Mask (`is_causal=True`): In a language model, a token's prediction must
    #     only depend on the tokens that came before it. The causal mask is a triangular
    #     matrix that sets the attention scores for all "future" tokens to negative
    #     infinity. After the softmax is applied, these scores become zero, ensuring
    #     the model cannot "cheat" by looking ahead in the sequence.
    #
    # 5.  Softmax: This function converts the raw, scaled attention scores into a
    #     probability distribution (i.e., they sum to 1). A token with a high score
    #     will have a high probability, indicating it is very important.
    #
    # 6.  Value Weighted Sum (@ V): The final output is a weighted sum of the Value
    #     vectors, where the weights are the probabilities calculated by the softmax.
    #     Tokens deemed important receive a higher weight, and their values contribute
    #     more to the output representation for the current token.
    #
    # Multi-Head Attention:
    # Instead of doing this once, we do it multiple times in parallel with different,
    # learned linear projections (n_head). Each "head" can focus on different aspects
    # of the relationships between tokens. The final outputs from all heads are then
    # concatenated and projected back to the original embedding dimension.
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
    # [INJECTOR: THE ANATOMY OF A TRANSFORMER BLOCK]
    #
    # The Transformer Block is the fundamental repeating unit of the GPT architecture.
    # It packages the two core computational modules—self-attention and the feed-forward
    # network—into a stable and effective structure.
    #
    # The two key architectural patterns employed here are:
    # 1.  Residual Connections (or "Shortcuts"): This is the `x + ...` pattern. The
    #     input to a module is added to its output. This technique is the single most
    #     important reason why we can train very deep neural networks. It combats the
    #     "vanishing gradient" problem by providing a direct, uninterrupted path for
    #     gradients to flow back through the network during backpropagation. Without
    #     residuals, a deep network's performance would degrade rapidly as more layers
    #     are added.
    #
    # 2.  Pre-Layer Normalization: Notice that Layer Normalization (`ln_1`, `ln_2`) is
    #     applied *before* the main operations (attention and MLP). This is a crucial
    #     design choice. Normalizing the input to each module stabilizes the training
    #     dynamics, leading to faster convergence and better overall performance
    #     compared to Post-Layer Normalization (where normalization is applied after
    #     the residual connection).
    #
    # The flow of data is as follows:
    #   input -> LayerNorm -> Attention -> + -> LayerNorm -> MLP -> + -> output
    #                                     |                        |
    #                                  (residual)               (residual)
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

        # [INJECTOR: THE ELEGANCE OF WEIGHT TYING]
        #
        # The token embedding layer (`wte`) and the final output layer (`lm_head`)
        # both operate on the same vocabulary. The `wte` maps a vocabulary index to
        # an embedding vector, and the `lm_head` maps an embedding vector back to a
        # logit over the vocabulary.
        #
        # Weight tying is a technique where we force these two layers to share the
        # same weight matrix. This has two primary benefits:
        # 1.  Parameter Reduction: It dramatically reduces the number of parameters
        #     in the model, as the largest weight matrices are now shared. This is
        #     especially significant for models with large vocabularies.
        # 2.  Improved Performance: Tying the weights imposes a useful constraint on
        #     the model, forcing it to learn a representation that is effective for
        #     both input and output. This often leads to better performance and
        #     faster convergence, as argued in papers like "Tying Word Vectors and
        #     Word Classifiers: A Loss Framework for Language Modeling."
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # [INJECTOR: THE SCIENCE OF WEIGHT INITIALIZATION]
        #
        # Proper weight initialization is crucial for stable training. In the GPT-2
        # paper ("Language Models are Unsupervised Multitask Learners"), the authors
        # propose a specific initialization scheme for the weights of the residual
        # projection layers (`c_proj`).
        #
        # The standard deviation is set to `0.02 / sqrt(2 * N)`, where `N` is the
        # number of residual layers (`n_layer`). This scaling accounts for the
        # accumulation of variance that occurs when adding multiple residual blocks
        # together. By scaling the initial weights down, we ensure that the output
        # of the residual blocks does not explode in magnitude at the beginning of
        # training, which would lead to instability.
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
