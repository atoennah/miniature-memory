# [INJECTOR: THE PHILOSOPHY OF A FROM-SCRATCH GPT]
#
# This module is the pedagogical core of a miniature GPT implementation. It is
# designed to be a transparent, educational asset for understanding the inner
# workings of a Transformer-based language model. Every component is intentionally
# kept explicit and self-contained to reveal the fundamental mechanics of modern
# large language models.
#
# The primary philosophical goals are:
#   1.  **Clarity over Brevity:** The code prioritizes readability and directness
#       over compact, obscure one-liners. The intent is to teach, not to obfuscate.
#   2.  **First Principles Implementation:** We avoid high-level abstractions where
#       they would hide the core logic (e.g., building self-attention from scratch).
#   3.  **Architectural Purity:** This model adheres to the canonical GPT-2/3
#       architecture, including the pre-LayerNorm configuration and causal
#       self-attention, to provide a stable reference point for learning.
#
# This file is not just code; it is a guided tour through the heart of a thinking
# machine. Read it, understand it, and build upon it.
#
# - Bolt ⚡

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
    # [INJECTOR: THE HIGH-LEVEL ARCHITECTURE OF THE GPT MODEL]
    #
    # This module orchestrates the entire GPT model, assembling all the core
    # components into a coherent, end-to-end language model. It defines the "macro"
    # architecture, detailing the flow of data from input tokens to output logits.
    #
    # ---
    #
    # **The Data Flow: From Integers to Probabilities**
    #
    # The `forward` method of this class tells the story of how the model processes
    # a sequence of token IDs (`idx`) to predict the next token.
    #
    #   1.  **Input Embeddings:** The process begins with two embedding layers:
    #       - `wte` (Word Token Embedding): This layer converts each integer token
    #         ID into a dense vector representation. This is the model's first
    #         step in moving from discrete symbols to a continuous, meaningful
    #         space.
    #       - `wpe` (Word Position Embedding): Since the self-attention mechanism
    #         is inherently position-agnostic, we must explicitly provide
    #         positional information. This layer creates a unique vector for each
    #         position in the sequence, which is then added to the token
    #         embedding. This allows the model to understand the order of words.
    #
    #   2.  **Transformer Blocks (`h`):** The summed token and positional embeddings
    #       are then passed through a deep stack of Transformer blocks. This is
    #       the computational core of the model where the deep learning happens.
    #       Each block refines the representations of the tokens by allowing them
    #       to attend to each other and undergo non-linear transformations.
    #
    #   3.  **Final Layer Normalization (`ln_f`):** After the final Transformer
    #       block, one last Layer Normalization is applied. This stabilizes the
    #       final representations before they are projected into the vocabulary space.
    #
    #   4.  **Language Model Head (`lm_head`):** This is a final linear layer that
    #       acts as a "decoder." It takes the high-level token representations
    #       from the Transformer and projects them into a very high-dimensional
    #       space, with one dimension for every word in the vocabulary. These
    #       raw, unnormalized outputs are called "logits."
    #
    #   5.  **Loss Calculation:** If `targets` are provided (during training), the
    #       logits are compared against the true next tokens using the
    #       `cross_entropy` loss function. This function measures how "surprised"
    #       the model was by the true next token. The goal of training is to
    #       minimize this surprise, thereby making the model's predictions more
    #       accurate. A softmax function is implicitly applied to the logits
    #       within `cross_entropy` to convert them into a probability
    #       distribution over the vocabulary.

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

class Block(nn.Module):
    # [INJECTOR: THE ARCHITECTURE OF A TRANSFORMER BLOCK]
    #
    # This module represents a single, self-contained block of a Transformer.
    # The architecture of a Transformer is essentially a deep stack of these
    # identical blocks. Each block is composed of two primary sub-layers:
    #
    #   1.  **Multi-Head Causal Self-Attention:** This is where the model performs
    #       the core "thinking" of relating different tokens in the sequence to
    #       each other.
    #
    #   2.  **Feed-Forward Network (MLP):** This is a standard multi-layer
    #       perceptron that processes each token's representation independently.
    #       It can be thought of as the "computation" or "reasoning" part of
    #       the block, where the model synthesizes the information gathered by
    #       the attention mechanism.
    #
    # ---
    #
    # **The Importance of Residual Connections and Layer Normalization**
    #
    # The true magic of deep architectures like the Transformer lies in two key
    # techniques employed around these sub-layers:
    #
    #   - **Residual Connections (`x = x + ...`):** This is the `+` operation in
    #     `x = x + self.attn(...)`. It creates a "shortcut" or "skip connection"
    #     that allows the gradient to flow directly through the network. This is
    #     absolutely critical for training very deep models. Without it, the
    #     gradients would diminish as they propagate backward through many layers
    #     (the "vanishing gradient" problem), making it impossible for the model
    #     to learn. The residual connection ensures that the model can learn an
    #     "identity function" by default, and only needs to learn the "residual"
    #     (the difference) that is beneficial.
    #
    #   - **Layer Normalization (`ln_1`, `ln_2`):** This is a normalization
    #     technique that is applied *before* each sub-layer. It normalizes the
    #     features for each token across the embedding dimension to have zero
    #     mean and unit variance. This stabilizes the learning process, reduces
    #     the sensitivity to the initialization of weights, and often speeds up
    #     training. The configuration used here (Pre-LN) is a common and robust
    #     variant.

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CausalSelfAttention(nn.Module):
    # [INJECTOR: THE THEORY OF CAUSAL SELF-ATTENTION]
    #
    # This module implements the mathematical core of the Transformer: Scaled
    # Dot-Product Attention, configured for autoregressive (causal) language
    # modeling.
    #
    # ---
    #
    # **The Core Formula: Q, K, V**
    #
    # At its heart, attention is a mechanism for a model to weigh the importance
    # of different tokens in a sequence when producing a representation for a
    # given token. It operates on three vectors derived from the input sequence:
    #
    #   - Q (Query): A representation of the current token, used to "ask" a
    #     question about other tokens.
    #   - K (Key): A representation of other tokens, used to be "queried" by Q.
    #   - V (Value): The actual information content of the other tokens.
    #
    # The similarity between a Query and a Key determines the "attention score."
    # This score is then used to create a weighted sum of the Values, effectively
    # allowing the model to focus on the most relevant parts of the input.
    #
    # ---
    #
    # **Scaled Dot-Product Attention**
    #
    # The specific formula used here is:
    #
    #   Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    #
    # Let's break this down:
    #
    #   1.  `Q @ K.T`: The dot product between the Query and Key vectors. This
    #       computes the raw similarity (or "attention") score between every
    #       pair of tokens. The result is a matrix of shape (T, T), where T is
    #       the sequence length.
    #
    #   2.  `/ sqrt(d_k)`: The scaling factor. `d_k` is the dimension of the Key
    #       vectors. This scaling is crucial. Without it, for large values of
    #       `d_k`, the dot products can grow very large in magnitude, pushing the
    #       softmax function into regions where its gradients are extremely
    #       small. This would kill the learning process. Scaling by the square
    #       root of the dimension keeps the variance of the dot products at 1,
    #       ensuring stable gradients.
    #
    #   3.  `softmax(...)`: This function is applied to the scaled scores. It
    #       normalizes the scores so that they all sum to 1, effectively
    #       turning them into a probability distribution of "attention weights."
    #
    #   4.  `@ V`: The final matrix multiplication. The attention weights are
    #       used to compute a weighted sum of the Value vectors. The result is
    #       an output vector that is a blend of the input tokens, weighted by
    #       their relevance to each other.
    #
    # ---
    #
    # **Causal (Masked) Attention**
    #
    # For language modeling, we need to ensure that the model is "autoregressive"
    # — meaning the prediction for the token at position `i` can only depend on
    # the known outputs at positions less than `i`. It cannot "look into the
    # future."
    #
    # This is achieved by "masking" the attention scores. Before the softmax
    # step, we add negative infinity (`-inf`) to all the elements in the
    # attention matrix that correspond to future positions. When softmax is
    # applied, these `-inf` values become zero, effectively preventing the model
    # from attending to future tokens. This is implemented via the `bias` buffer.
    #
    # ---
    #
    # **Multi-Head Attention**
    #
    # Instead of performing a single attention calculation, we split the embedding
    # dimension (`n_embd`) into multiple "heads" (`n_head`). Each head performs
    # its own independent attention calculation on a smaller portion of the
    # embedding. The results from all heads are then concatenated and projected
    # back to the original embedding size.
    #
    # This allows the model to jointly attend to information from different
    # representational subspaces at different positions. For example, one head
    # might focus on syntactic relationships, while another focuses on semantic
    # meaning.

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    # [INJECTOR: THE ROLE OF THE FEED-FORWARD NETWORK (MLP)]
    #
    # This module implements the Feed-Forward Network (FFN), the second major
    # component of a Transformer block (after the attention mechanism). While the
    # attention mechanism is responsible for communication and information routing
    # between tokens, the MLP is responsible for the actual computation and
    # transformation of the information for each token independently.
    #
    # ---
    #
    # **Architectural Design**
    #
    # The MLP consists of two linear layers with a non-linear activation function
    # in between.
    #
    #   1.  `c_fc`: The first linear layer, which expands the dimensionality of the
    #       input embedding (`n_embd`) by a factor of 4. This expansion creates
    #       a higher-dimensional "thinking space" for the model to process the
    #       information. The expansion factor of 4 is a standard, empirically-
    #       derived choice from the original Transformer paper ("Attention Is All
    #       You Need").
    #
    #   2.  `gelu`: The GELU (Gaussian Error Linear Unit) activation function.
    #       Unlike simpler activations like ReLU, GELU provides a smoother,
    #       probabilistically-inspired non-linearity. It weights inputs by their
    #       magnitude, which has been shown to be highly effective in
    #       Transformer models.
    #
    #   3.  `c_proj`: The second linear layer, which projects the high-dimensional
    #       representation back down to the original embedding dimension. This
    #       "projection" step finalizes the computation for the block.
    #
    # In essence, the MLP allows the model to process the rich, context-aware
    # representations generated by the attention mechanism and extract higher-level
    # features, before passing the result to the next block or the final output.

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
