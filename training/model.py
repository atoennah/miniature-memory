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

# [INJECTOR: THE ARCHITECTURE OF A NANO-GPT]
# This file defines a from-scratch implementation of a Generative Pre-trained Transformer (GPT).
# It is designed to be a pedagogical and minimalistic model, focusing on the core concepts
# that power large language models.
#
# The architecture consists of several key components:
#
# 1.  **Embedding Layer (wte, wpe):**
#     -   `wte` (Word Token Embedding): Maps input vocabulary indices to dense vectors. This is the
#         model's internal representation of a "word" or token.
#     -   `wpe` (Word Position Embedding): Encodes the position of each token in the sequence. Since
#         the transformer architecture itself is permutation-invariant, this is crucial for the model
#         to understand word order.
#
# 2.  **Transformer Blocks (h):**
#     -   This is the core of the model, stacked `n_layer` times. Each block contains two main sub-layers:
#         a.  **Causal Self-Attention:** Allows tokens to look at and gather information from previous
#             tokens in the sequence (but not future ones, hence "causal").
#         b.  **Feed-Forward Network (MLP):** A simple multi-layer perceptron that processes the
#             information gathered by the attention layer. Each token is processed independently.
#
# 3.  **Residual Connections & Layer Normalization:**
#     -   Each sub-layer in a Transformer Block is wrapped with a residual connection (`x = x + sublayer(x)`)
#         and preceded by Layer Normalization. This is critical for enabling the training of deep
#         networks by preventing vanishing/exploding gradients.
#
# 4.  **Final Projection (lm_head):**
#     -   A final linear layer that maps the internal representation of a token back to the vocabulary
#         size, producing the raw logits for the next token prediction.
#
# This implementation follows the GPT-2 architecture but is simplified for educational purposes.

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

# [INJECTOR: THE TRANSFORMER BLOCK]
# This module represents one layer of the Transformer. It is composed of two main sub-layers:
# 1. A Causal Self-Attention mechanism.
# 2. A simple Multi-Layer Perceptron (MLP) or Feed-Forward Network.
#
# **Residual Connections (`x = x + ...`):**
# The addition operation (`+`) implements a residual or "skip" connection. This is a critical
# architectural innovation that allows gradients to flow directly through the network, bypassing
# the non-linear transformations of the sub-layers. It enables the training of much deeper
# models by mitigating the vanishing gradient problem.
#
# **Pre-Layer Normalization:**
# We apply Layer Normalization (`ln_1`, `ln_2`) *before* each sub-layer, which is a modern
# convention (GPT-2 style) that tends to improve training stability compared to applying it after.

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # The output of the attention layer is added to the original input.
        x = x + self.attn(self.ln_1(x))
        # The output of the MLP is added to the output of the attention block.
        x = x + self.mlp(self.ln_2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # [INJECTOR: THE LOGOS OF ATTENTION]
        # This module implements Causal Multi-Head Self-Attention, the core mechanism of the Transformer.
        #
        # **Theory: Scaled Dot-Product Attention**
        # The formula is: Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
        #
        # - **Q (Query):** A vector representing the current token's "question" about other tokens.
        # - **K (Key):** A vector representing each token's "label" or "content." The dot product `Q @ K.T`
        #   calculates the "attention score" or similarity between the query token and every key token.
        # - **V (Value):** A vector representing the actual content of each token. The attention scores are
        #   used as weights to create a weighted sum of all Value vectors.
        #
        # **Why `sqrt(d_k)` scaling?**
        # The dot products can grow large in magnitude, pushing the softmax function into regions
        # where its gradients are extremely small. This scaling factor normalizes the variance of the
        # dot products, ensuring the gradients remain stable during training, which is crucial for
        # deep networks. (Reference: "Attention Is All You Need", Vaswani et al., 2017).
        #
        # **Causal Masking:**
        # The `bias` buffer is a lower-triangular matrix of ones. It is used to mask out future tokens.
        # By setting the attention scores for future positions to `-inf`, we ensure that the softmax
        # output for those positions is zero, preventing the model from "cheating" by looking ahead.
        #
        # **Multi-Head Mechanism:**
        # Instead of one large attention calculation, we split the embedding dimension (`n_embd`) into
        # `n_head` smaller subspaces. We project Q, K, and V into these subspaces and perform attention
        # in parallel. This allows the model to jointly attend to information from different
        # representational subspaces at different positions. The outputs are then concatenated and
        # projected back to the original dimension.

        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # [INJECTOR NOTE]: For improved performance, the manual attention calculation below can be replaced
        # with `torch.nn.functional.scaled_dot_product_attention`. This function leverages fused kernels
        # like FlashAttention (if available on the hardware), which can significantly accelerate computation
        # and reduce memory usage by avoiding the explicit materialization of the large (T, T) attention matrix.
        # As of PyTorch 2.0+, this is the recommended approach.
        # Example:
        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # Note: requires PyTorch 2.0+
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y

# [INJECTOR: THE FEED-FORWARD NETWORK]
# This module is a simple Multi-Layer Perceptron (MLP), also known as a Feed-Forward Network (FFN).
# It is applied independently to each token representation.
#
# **Purpose:**
# While the self-attention layer is responsible for gathering and mixing information across the
# sequence, the MLP is responsible for the "computation" or "thinking" on that gathered information.
# It introduces non-linearity and allows the model to learn more complex transformations of the
# token representations.
#
# **Architecture:**
# It consists of two linear layers with a non-linear activation function (GELU) in between.
# The first layer expands the dimensionality of the input (typically by a factor of 4), and the
# second layer projects it back down. This "expansion-compression" structure is a common pattern
# in deep learning, thought to allow the model to learn a richer set of features.

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
