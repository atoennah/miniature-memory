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

class GPT(nn.Module):
    """A GPT-style transformer model."""
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
    block_size: int = 256
    vocab_size: int = 65 # Default: char-level for English text
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2

class MultiHeadAttention(nn.Module):
    # [INJECTOR: MERGED IMPLEMENTATION - ATTENTION LOGOS & PARALLEL REALITIES]
    #
    # This module combines the logic of single-head attention and multi-head attention
    # into a single, optimized implementation that leverages PyTorch's fused kernels.
    #
    # --- The Logos of Self-Attention ---
    # The core formula remains: Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    # Q (Query): What I am looking for.
    # K (Key):   What I contain.
    # V (Value): What I will give you.
    #
    # --- The "Parallel Realities" of Multi-Head Attention ---
    # Instead of iterating through separate `Head` modules, we perform the projections for
    # all heads simultaneously using a single `c_attn` linear layer. This is highly efficient.
    # The output of this layer (3 * n_embd) is then split and reshaped to create the Q, K, V
    # tensors for all heads at once.
    #
    # --- Fused Kernel Optimization ---
    # The entire sequence of scaling, masking, softmax, and value-weighting is performed
    # in a single call to `torch.nn.functional.scaled_dot_product_attention`. This function
    # is highly optimized to use hardware-specific kernels like FlashAttention when available.
    # The `is_causal=True` argument replaces the manual lower-triangular mask (`tril`),
    # ensuring that the model remains autoregressive.
    """Multiple heads of self-attention in parallel"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Special scaling for residual connections, per NanoGPT
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # Regularization
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # Flash attention is implemented in PyTorch 2.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

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
        # A single linear layer projects the input into Q, K, and V for all heads at once.
        # This is more efficient than creating separate linear layers for each head.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # A final projection layer to combine the outputs of all heads.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd


    def forward(self, x):
        B, T, C = x.shape # Batch size, Sequence length, Embedding dimensionality (n_embd)

        # 1. Project to Q, K, V for all heads in batch.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 2. Reshape and transpose Q, K, V to prepare for multi-head attention.
        #    The shape changes from (B, T, C) to (B, n_head, T, head_size).
        #    This brings the head dimension forward, allowing each head to perform attention independently.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 3. Perform scaled dot-product attention using the fused kernel.
        #    - `is_causal=True` handles the causal masking automatically.
        #    - Dropout is applied internally if the model is in training mode.
        #    - The scaling by 1/sqrt(head_size) is also handled internally.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        # 4. Re-assemble all head outputs side-by-side.
        #    The transpose and contiguous().view() operations reverse the reshaping from step 2,
        #    concatenating the head outputs to get a final shape of (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5. Apply the final output projection.
        y = self.dropout(self.c_proj(y))
        return y
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        # is_causal=True will automatically apply the lower-triangular mask
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)

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
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for the GPT model.

        Args:
            idx: A tensor of shape (B, T) containing the input token indices.
            targets: An optional tensor of shape (B, T) containing the target token indices.

        Returns:
            A tuple containing:
                - logits: A tensor of shape (B, T, vocab_size) containing the model's predictions.
                - loss: An optional tensor containing the cross-entropy loss.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # Token and position embeddings
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    def forward(self, x):
        # First sub-layer: Self-Attention, preceded by LayerNorm, followed by a residual connection.
        x = x + self.sa(self.ln1(x))
        # Second sub-layer: Feed-Forward, preceded by LayerNorm, followed by a residual connection.
        x = x + self.ffwd(self.ln2(x))
        return x
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.h:
            x = block(x)

        # Final layer norm and language model head
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

class Block(nn.Module):
    """A single transformer block."""
    def __init__(self, config: GPTConfig):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a transformer block."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CausalSelfAttention(nn.Module):
    """A causal self-attention module."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Regularization
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
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the causal self-attention module."""
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # Output projection
        # ⚡ Bolt: Replaced manual attention with fused scaled_dot_product_attention
        # This leverages Flash Attention kernels for significant speedup and memory savings.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """A simple multi-layer perceptron."""
    def __init__(self, config: GPTConfig):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLP."""
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
