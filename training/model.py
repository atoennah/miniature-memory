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

class Head(nn.Module):
    # [INJECTOR: THE LOGOS OF SELF-ATTENTION]
    #
    # This module implements a single head of Scaled Dot-Product Attention, the fundamental
    # building block of a Transformer. Its purpose is to allow tokens in a sequence to
    # communicate with each other, weighing their importance.
    #
    # The core formula is: Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    #
    # Q (Query): What I am looking for. A projection of the current token's embedding.
    # K (Key):   What I contain. A projection of another token's embedding.
    # V (Value): What I will give you. A projection of another token's embedding.
    #
    # The dot product (Q @ K.T) measures the similarity or "affinity" between the Query
    # of one token and the Key of another. A high dot product means the tokens are highly relevant.
    """One head of self-attention"""

    def __init__(self, config: GPTConfig, head_size: int):
        super().__init__()
        # Linear projections for Query, Key, and Value.
        # These learnable weights transform the input embedding (n_embd) into the attention space (head_size).
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)

        # [INJECTOR NOTE: THE CAUSAL MASK]
        # The 'tril' buffer is not a learnable parameter. It's a lower-triangular matrix of ones.
        # Its purpose is to enforce causality in the self-attention mechanism. In an autoregressive
        # model, a token at position `i` should only attend to tokens at positions `j <= i`.
        # By masking out the upper triangle of the affinity matrix with -infinity before the softmax,
        # we ensure that future tokens have zero influence on the current token's output.
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape # Batch, Time (sequence length), Channels (embedding dimension)

        # 1. Project input into Query, Key, and Value spaces.
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # 2. Compute attention scores ("affinities").
        #    q @ k.T results in a (B, T, T) matrix where wei[b, i, j] is the affinity
        #    between the i-th token's query and the j-th token's key in the b-th batch.
        #
        # [INJECTOR NOTE: THE SCALING FACTOR]
        # We scale by C**-0.5, which is 1/sqrt(d_k) where d_k is the head dimension.
        # Why? As the embedding dimension C grows, the variance of the dot product Q @ K.T
        # also grows. Large dot products push the softmax function into regions where its
        # gradients are vanishingly small, making learning unstable. The scaling factor
        # counteracts this by normalizing the variance, ensuring stable gradients.
        # See paper "Attention Is All You Need" (Vaswani et al., 2017), Section 3.2.1.
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T)

        # 3. Apply the causal mask.
        #    Replace all values in the upper triangle (where the model would be "looking ahead")
        #    with negative infinity. The softmax will turn these into zeros.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)

        # 4. Normalize scores into a probability distribution (weights).
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # 5. Compute the weighted sum of Values.
        #    Each output token embedding is a sum of all previous value vectors,
        #    weighted by their attention scores.
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, head_size)

        return out

class MultiHeadAttention(nn.Module):
    # [INJECTOR: THE "PARALLEL REALITIES" OF MULTI-HEAD ATTENTION]
    #
    # While a single attention head can learn one type of relationship between tokens (e.g.,
    # focusing on the preceding verb), it's beneficial for the model to learn several
    # different types of relationships simultaneously.
    #
    # Multi-Head Attention achieves this by running multiple `Head` modules in parallel.
    # Each head operates on a smaller subspace of the embedding dimension (`head_size`).
    # For example, if `n_embd` is 384 and `n_head` is 6, each head will process a `head_size` of 64.
    #
    # This is analogous to having multiple experts look at the same sentence:
    # - Head 1 might learn to track syntactic dependencies.
    # - Head 2 might learn to identify co-reference (which pronouns refer to which nouns).
    # - Head 3 might learn to capture semantic similarity.
    #
    # By concatenating the outputs of these parallel heads, we create a rich representation
    # that captures a diverse set of relationships. The final linear projection (`proj`)
    # learns to integrate these parallel realities back into a single, coherent embedding.
    """Multiple heads of self-attention in parallel"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd) # Projection back to the residual pathway
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # 1. Run each attention head in parallel.
        #    h(x) for h in self.heads results in a list of tensors, each of shape (B, T, head_size).
        # 2. Concatenate the outputs along the channel dimension.
        #    This combines the "perspectives" of all heads into a single tensor of shape (B, T, n_embd).
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # 3. Project the concatenated outputs back into the main embedding space.
        #    This allows the network to learn how to best combine the information from the different heads.
        out = self.dropout(self.proj(out))
        return out

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
