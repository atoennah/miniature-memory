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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the causal self-attention module."""
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # [INJECTOR: DEMYSTIFYING MULTI-HEAD TENSOR MANIPULATION]
        #
        # The core idea of multi-head attention is to run the attention mechanism in
        # parallel several times, with different, learned linear projections for Q, K,
        # and V. This allows the model to jointly attend to information from different
        # representational subspaces at different positions.
        #
        # The tensor transformations below are the key to making this efficient.
        # Let's break down the journey of the Key tensor `k`:
        #
        # 1.  Initial shape of `k`: `(B, T, C)`
        #     - B = Batch size (number of sequences processed at once)
        #     - T = Sequence length (e.g., `block_size`)
        #     - C = Embedding dimension (`n_embd`)
        #
        # 2.  `k.view(B, T, self.n_head, C // self.n_head)`
        #     - This reshapes the tensor without changing its data. We are splitting the
        #       embedding dimension `C` into `n_head` smaller chunks.
        #     - `hs = C // self.n_head` is the "head size".
        #     - New shape: `(B, T, nh, hs)` where `nh` is `n_head`.
        #     - This logically groups the embeddings for each head, but they are still
        #       interleaved in memory.
        #
        # 3.  `.transpose(1, 2)`
        #     - This is the crucial step. We swap the sequence length dimension (T) with
        #       the number of heads dimension (nh).
        #     - New shape: `(B, nh, T, hs)`
        #     - Why? The `scaled_dot_product_attention` function expects the heads to
        #       be in the "batch" dimension. By rearranging the tensor this way, we
        #       create a batch of `B * nh` attention problems, each of size `(T, hs)`.
        #       PyTorch's optimized kernel can then process all these heads in parallel,
        #       which is massively faster than looping through them.
        #
        # The same transformation is applied to `q` and `v`, preparing them for the
        # batched attention calculation.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention using PyTorch's fused kernel
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        # [INJECTOR: MANUAL IMPLEMENTATION FOR PEDAGOGY]
        #
        # For educational clarity, the manual implementation of causal self-attention
        # is presented below. The fused `F.scaled_dot_product_attention` is highly
        # optimized and should be used in practice, but understanding the underlying
        # operations is crucial.
        #
        # B, nh, T, hs = q.shape
        #
        # # 1. Calculate attention scores ("affinities")
        # # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #
        # # 2. Apply causal mask
        # # `tril` creates a lower-triangular matrix. `masked_fill` replaces all
        # # elements where the condition is True (i.e., the upper triangle) with -inf.
        # mask = torch.tril(torch.ones(T, T, device=q.device)).view(1, 1, T, T)
        # att = att.masked_fill(mask == 0, float('-inf'))
        #
        # # 3. Apply softmax to get weights
        # att = F.softmax(att, dim=-1)
        #
        # # 4. Apply dropout
        # att = F.dropout(att, p=self.dropout, training=self.training)
        #
        # # 5. Weighted sum of values
        # # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        # y = att @ v
        #
        # The result `y` from this manual calculation would be identical in shape and
        # (approximately) in value to the output of the fused kernel.

        # Re-assemble all head outputs side by side
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    # [INJECTOR: THE ARCHITECTURE OF A TRANSFORMER BLOCK]
    #
    # A Transformer is essentially a deep stack of these "Block" modules. The depth is
    # what allows the model to build up progressively more abstract and complex
    # representations of the input text. However, training very deep neural networks is
    # notoriously difficult due to the "vanishing gradient" problem.
    #
    # Two key architectural innovations in this block make it possible:
    #
    # 1.  Residual Connections (The `+` in `x + self.attn(...)`):
    #     This is also known as a "skip connection". Instead of forcing the output of a
    #     sub-layer (like attention or the MLP) to represent the *entire* desired output,
    #     we only ask it to learn the *residual* or the *change* from the input.
    #     The input `x` is passed directly through (the "skip") and added to the output
    #     of the sub-layer. This creates a direct, unimpeded path for gradients to flow
    #     back through the network during backpropagation. Without this, gradients would
    #     have to flow through many layers of non-linear transformations, diminishing
    #     at each step until they become too small to effectively update the weights of
    #     the earliest layers.
    #
    # 2.  Pre-Layer Normalization (The `ln_1` and `ln_2` before Attention/MLP):
    #     Layer Normalization stabilizes the training process by normalizing the inputs
    #     to each sub-layer. It ensures that the mean of the inputs is 0 and the standard
    #     deviation is 1. By placing it *before* the main operations (Attention and MLP),
    #     we adopt the "Pre-LN" architecture. This has been found to be more stable and
    #     effective for training very deep Transformers than the original "Post-LN"
    #     variant where normalization was applied after the residual connection.
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
            # [INJECTOR: THE NECESSITY OF POSITIONAL EMBEDDINGS]
            #
            # The self-attention mechanism is permutation-invariant; it treats the input
            # as a "bag of tokens" with no inherent order. "The cat sat on the mat" and
            # "The mat sat on the cat" would produce nearly identical attention scores
            # if only token embeddings were used.
            #
            # To solve this, we inject positional information directly into the model.
            # The `wpe` (Word Position Embedding) layer is a learnable lookup table where
            # each position in the sequence (from 0 to `block_size - 1`) has a unique
            # vector. This positional vector is added to the token embedding.
            #
            # tok_emb (B, T, C) + pos_emb (1, T, C) -> input (B, T, C)
            #
            # By adding these two embeddings, we create a composite representation that
            # encodes both *what* a token is (its semantic meaning) and *where* it is
            # in the sequence. The model can then learn to use this positional information
            # to understand grammar, syntax, and long-range dependencies.
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # [INJECTOR: THE EFFICIENCY OF WEIGHT TYING]
        #
        # In many Transformer-based language models, the token embedding matrix (which maps
        # vocabulary indices to high-dimensional vectors) and the final output linear layer
        # (which maps the model's internal representation back to vocabulary logits) have
        # the same dimensions: `vocab_size` x `n_embd`.
        #
        # Weight tying is the practice of forcing these two matrices to be the same.
        # `self.transformer.wte.weight` and `self.lm_head.weight` will point to the same
        # block of memory and will be updated together during training.
        #
        # Why do this?
        # 1.  Parameter Efficiency: This is the most significant benefit. It dramatically
        #     reduces the number of parameters in the model, as the `lm_head` no longer
        #     needs its own large weight matrix. For a vocabulary of 50,000 and an
        #     embedding size of 768, this saves ~38 million parameters.
        # 2.  Improved Performance: The shared weights create a more direct relationship
        #     between how a token is represented on input and how its probability is
        #     calculated on output. This has been shown to improve the performance of
        #     language models. The intuition is that if the model "knows" that the input
        #     and output layers are linked, it learns more meaningful embeddings.
        # Link: https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # Pre-compute positional indices and register as a buffer
        pos = torch.arange(0, config.block_size, dtype=torch.long).unsqueeze(0)
        self.register_buffer('pos', pos, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # [INJECTOR: GPT-2 STYLE WEIGHT INITIALIZATION]
        #
        # The original GPT-2 paper found that a special initialization scheme for the weights
        # of the residual projection layers (`c_proj`) was beneficial for training.
        #
        # The standard initialization (`std=0.02`) is used for most layers. However, for
        # the projection layers that are part of the residual path, the standard deviation
        # is scaled down by a factor of `sqrt(2 * N)`, where `N` is the number of
        # transformer layers (`n_layer`).
        #
        # Why this scaling?
        # At initialization, the residual connections ensure that the model starts as an
        # identity function. The outputs of the attention and MLP blocks are initially
        # very small and are added to the input. This scaling prevents the outputs of
        # these residual blocks from being too large at the beginning of training, which
        # could destabilize the learning process. By scaling down the initial weights, we
        # ensure that the model learns cautiously, gradually building upon the identity
        # function provided by the skip connections.
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
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = self.pos[:, :t]

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

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_p: float = 0.9) -> torch.Tensor:
        # [INJECTOR: THE ART & SCIENCE OF DECODING STRATEGIES]
        #
        # Generating text from a language model is a creative process, and the choice
        # of decoding strategy profoundly impacts the quality and nature of the output.
        #
        # 1.  Greedy Decoding (The Simplest):
        #     - At each step, simply choose the token with the highest probability (logit).
        #     - Problem: It's deterministic and often leads to repetitive, boring, and
        #       predictable text. It can easily get stuck in loops.
        #
        # 2.  Temperature Sampling:
        #     - We introduce a `temperature` parameter to control the "randomness" of the
        #       output. The logits are divided by the temperature before the softmax.
        #     - `temperature < 1.0`: Makes the distribution sharper, increasing the
        #       likelihood of high-probability tokens. The output becomes more focused
        #       and deterministic, closer to greedy decoding.
        #     - `temperature > 1.0`: Makes the distribution flatter, increasing the
        #       likelihood of low-probability tokens. The output becomes more surprising
        #       and creative, but also risks being nonsensical.
        #
        # 3.  Top-P (Nucleus) Sampling (The Goldilocks Zone):
        #     - This is the strategy implemented here, and it's a significant improvement
        #       over the other two. Instead of considering all tokens or just the top one,
        #       it focuses on the smallest possible set of tokens whose cumulative
        #       probability exceeds a threshold `top_p`.
        #     - How it works:
        #         a. Sort the probabilities in descending order.
        #         b. Sum them up cumulatively.
        #         c. Find the point where the cumulative sum exceeds `top_p`.
        #         d. All tokens *after* this point are discarded.
        #         e. The remaining tokens (the "nucleus") are renormalized and sampled from.
        #     - Why it's better: It's adaptive. For "high-confidence" predictions where
        #       the model is sure about the next token (e.g., after "President of the
        #       United States, Joe..."), the nucleus will be very small, perhaps just
        #       {"Biden"}. For "low-confidence" predictions where many words could fit,
        #       the nucleus will be larger, allowing for more creativity. This avoids the
        #       pitfalls of temperature sampling, which can produce gibberish by giving
        #       too much weight to unlikely tokens.
        """
        Autoregressively generates a sequence of tokens using top-p (nucleus) sampling.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            # Apply temperature scaling.
            logits = logits[:, -1, :] / temperature

            # [INJECTOR: STEP-BY-STEP NUCLEUS SAMPLING]

            # 1. Convert logits to probabilities via softmax.
            probs = F.softmax(logits, dim=-1)

            # 2. Sort probabilities in descending order and get their original indices.
            # This is the core of identifying the "nucleus" of the most likely tokens.
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)

            # 3. Calculate the cumulative sum of sorted probabilities. This will allow us
            # to find the smallest set of tokens whose cumulative probability is >= top_p.
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # 4. Create a mask to identify tokens to remove.
            # `cumulative_probs > top_p` gives us all tokens *after* the nucleus.
            sorted_indices_to_remove = cumulative_probs > top_p

            # 5. Important edge case: we must keep at least one token. We shift the mask
            # to the right, ensuring that the first token (the most likely one) is never
            # removed, even if its probability alone exceeds top_p.
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # 6. Get the original indices of the tokens to be removed by gathering from
            # the `sorted_indices` tensor.
            indices_to_remove = sorted_indices[sorted_indices_to_remove]

            # 7. Zero out the probabilities of the tokens to be removed.
            probs[:, indices_to_remove] = 0

            # 8. Renormalize the probability distribution so that the remaining tokens
            # (the nucleus) sum to 1.
            probs = probs / torch.sum(probs, dim=-1, keepdim=True)

            # 9. Sample the next token from the renormalized distribution.
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx
