# [INJECTOR: THE PHILOSOPHY OF A FROM-SCRATCH GPT]
#
# This module is the pedagogical core of a miniature GPT implementation. Its purpose is not just to work,
# but to teach. Every line is a deliberate choice to expose the fundamental mechanics of the Transformer
# architecture as laid out in the original "Attention Is All You Need" paper (Vaswani et al., 2017).
#
# We intentionally avoid high-level abstractions where they would obscure the underlying logic. For instance,
# instead of a monolithic `nn.TransformerEncoderLayer`, we build it from scratch using LayerNorm,
# CausalSelfAttention, and a Multi-Layer Perceptron (MLP). This reveals the data flow and the purpose
# of each component, particularly the residual connections that are critical for training deep networks.
#
# This implementation is designed for:
#   1.  **Clarity:** To be read, understood, and modified by students of deep learning.
#   2.  **Simplicity:** To demonstrate that a powerful language model can be built from a few core,
#       comprehensible components.
#   3.  **Correctness:** To faithfully implement the key architectural details, such as causal masking
#       for autoregressive generation and scaled dot-product attention.
#
# Read this code not just as a script, but as a textbook. Each class is a chapter, each function a concept.
#
# [REFERENCE]: https://arxiv.org/abs/1706.03762

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

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # [IMPLEMENTATION NOTE]: The model is composed of three main parts:
        # 1. An embedding layer for tokens (`wte`) and positions (`wpe`).
        # 2. A stack of `n_layer` transformer blocks (`h`).
        # 3. A final layer norm (`ln_f`) and a linear layer (`lm_head`) to produce logits.
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

        # 1. --- Embeddings ---
        # Each token index `idx` is mapped to a vector (`tok_emb`).
        # Each position `pos` is also mapped to a vector (`pos_emb`).
        # The two are summed to create a position-aware token representation.
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        # 2. --- Transformer Blocks ---
        # The input `x` is processed sequentially by each block in the stack.
        # Each block applies self-attention and an MLP, with residual connections.
        for block in self.h:
            x = block(x)

        # 3. --- Final Layers ---
        # The final layer norm stabilizes the activations before the final projection.
        # The linear head (`lm_head`) projects the final transformer output to the vocabulary size,
        # producing the raw logits for the next token prediction.
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # The cross-entropy loss compares the predicted logits with the true next-token `targets`.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

class Block(nn.Module):
    # [INJECTOR: THE ANATOMY OF A TRANSFORMER BLOCK]
    #
    # A Transformer Block is the repeating unit of the GPT architecture. It has two main sub-layers:
    #   1. A Multi-Head Causal Self-Attention layer.
    #   2. A position-wise Feed-Forward Network (an MLP in this case).
    #
    # Crucially, each sub-layer is wrapped with two architectural patterns:
    #   a. Pre-Layer Normalization: Normalization is applied *before* the main operation (attention or MLP).
    #      This tends to lead to more stable training than post-layer norm.
    #   b. Residual Connections: The input to the sub-layer is added to its output (`x = x + sublayer(ln(x))`).
    #      This is the key to training very deep networks. It creates a "shortcut" for the gradient to
    #      flow through, mitigating the vanishing gradient problem.

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # The forward pass follows the "Pre-LayerNorm" structure.
        # x -> LayerNorm -> Attention -> (+) -> x -> LayerNorm -> MLP -> (+) -> x
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CausalSelfAttention(nn.Module):
    # [INJECTOR: THE LOGOS OF SELF-ATTENTION]
    #
    # This module implements Causal Self-Attention, the core mechanism of the Transformer decoder.
    # At a high level, attention allows a model to weigh the importance of different tokens in the input
    # sequence when producing a representation for a given token. "Self-Attention" means the sequence
    # attends to itself. "Causal" (or "masked") means that a token at position `i` can only attend to
    # tokens at positions `j <= i`. This is crucial for autoregressive models that generate text one
    # token at a time.
    #
    # The mathematical formulation is Scaled Dot-Product Attention:
    #
    #   Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
    #
    #   - Q (Query): A projection of the current token's representation. It "asks" a question.
    #   - K (Key): A projection of all tokens' representations. It represents what each token "offers."
    #   - V (Value): A projection of all tokens' representations. It's the content that gets aggregated.
    #
    # The dot product `Q * K^T` computes a similarity score between the query and each key.
    # The scaling factor `1 / sqrt(d_k)` (where d_k is the dimension of K) is vital. Without it,
    # for large `d_k`, the dot products can grow very large, pushing the softmax into regions with
    # extremely small gradients, which harms learning. This is a critical and subtle implementation detail.
    #
    # Multi-Head Attention:
    # Instead of one large attention calculation, we split the embedding dimension `n_embd` into `n_head`
    # smaller subspaces ("heads"). Attention is computed independently in each head, and the results are
    # concatenated. This allows the model to jointly attend to information from different representational
    # subspaces at different positions. It's like having multiple experts look at the same sentence from
    # different perspectives.

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # [IMPLEMENTATION NOTE]: A single linear layer projects the input `x` to Q, K, and V matrices simultaneously.
        # This is an efficient optimization. `3 * config.n_embd` is for the concatenation of Q, K, V.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # This is the output projection layer, after the attention values have been aggregated.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # [IMPLEMENTATION NOTE]: The causal mask (`bias`) is registered as a buffer, not a parameter.
        # This means it's part of the model's state but is not considered a trainable parameter.
        # `torch.tril` creates a lower-triangular matrix, ensuring a token at position `i` can only
        # attend to tokens at positions `j <= i`.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence length, Embedding dimensionality (n_embd)

        # 1. --- Project to Q, K, V ---
        # (B, T, C) -> (B, T, 3 * C) -> split into 3 * (B, T, C)
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        # 2. --- Reshape for Multi-Head Attention ---
        # (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        # where head_size = C // n_head.
        # The transpose is crucial: it brings the head dimension to the forefront, so the matrix multiplication
        # for attention is performed independently for each head.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 3. --- Scaled Dot-Product Attention ---
        # Q * K^T -> (B, n_head, T, head_size) @ (B, n_head, head_size, T) -> (B, n_head, T, T)
        # The result is an "attention score" matrix for each head.
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
        # Apply the causal mask. `masked_fill` replaces all positions where the mask is 0 with -infinity.
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # Softmax normalizes the scores along the key dimension (dim=-1), turning them into probabilities.
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # Aggregate the values based on the attention scores.
        # (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)
        y = att @ v

        # 4. --- Concatenate Heads and Project Output ---
        # The `transpose` and `contiguous().view()` operations reverse the reshaping from step 2,
        # effectively concatenating the heads' outputs.
        # (B, n_head, T, head_size) -> (B, T, n_head, head_size) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection.
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
