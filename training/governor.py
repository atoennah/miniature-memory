import torch
import math
from typing import Tuple, List, Optional
from dataclasses import dataclass

class LinguisticGovernor:
    """
    [BOLT: THE LINGUISTIC GOVERNOR - RECONSTRUCTED]
    A modular implementation of advanced penalty mechanisms for Indonesian GPT.
    Handles Entropy, Repetition, Contextual, and Phonotactic constraints.
    """
    def __init__(self, config, vocab_size: int, device: str):
        self.config = config
        self.vocab_size = vocab_size
        self.device = device

        # Prepare buffers
        if config.vowel_ids:
            self.register_buffer('vowel_ids', torch.tensor(config.vowel_ids, device=device))
        if config.consonant_ids:
            self.register_buffer('consonant_ids', torch.tensor(config.consonant_ids, device=device))

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def calculate_penalty(self, logits: torch.Tensor, xb: torch.Tensor, step: int) -> torch.Tensor:
        B, T, C = logits.size()
        probs = torch.softmax(logits, dim=-1)

        # 1. Entropy & Length-Normalized Weighted Penalty
        entropy_per_pos = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1) # (B, T)
        pos_weights = torch.linspace(0.5, 1.5, T, device=self.device).view(1, T)

        # 2. Contextual Penalty Scaling (Word Boundary Relief)
        ctx_weights = torch.ones_like(entropy_per_pos)
        is_space = (xb == self.config.space_token_id)
        ctx_weights[is_space] = 0.5

        # 3. N-Gram Repetition Penalty (Memory Optimized)
        rep_penalty = torch.tensor(0.0, device=self.device)
        if self.config.repetition_penalty > 0:
            # Instead of one-hot scatter_, we use gather to get probabilities at xb indices
            # xb[:, i] is the token at position i. logits[:, i] predicts token i+1.
            # We want to penalize if the model predicts token i at position i (stuttering).
            # Alignment: prediction at index i is for target yb[i] which corresponds to xb[i+1].
            # To penalize predicting xb[i] at index i:
            gathered_probs = torch.gather(probs, 2, xb.unsqueeze(2)).squeeze(2)
            rep_penalty = self.config.repetition_penalty * (gathered_probs * ctx_weights).mean()

        # 4. Phonotactic Constraint Scaling (CV-Ratio)
        phono_penalty = torch.tensor(0.0, device=self.device)
        if (self.config.phonotactic_penalty > 0 and
            hasattr(self, 'vowel_ids') and hasattr(self, 'consonant_ids')):

            # Apply Cosine Decay
            current_phono_lambda = self.config.phonotactic_penalty
            if step > self.config.phono_decay_start:
                decay_step = step - self.config.phono_decay_start
                if decay_step < self.config.phono_decay_iters:
                    decay_ratio = decay_step / self.config.phono_decay_iters
                    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                    current_phono_lambda *= coeff
                else:
                    current_phono_lambda = 0.0

            # Cluster Detection (Optimized with masking)
            v_mask = torch.isin(xb, self.vowel_ids).float()
            c_mask = torch.isin(xb, self.consonant_ids).float()
            v_cluster = (v_mask[:, :-1] * v_mask[:, 1:])
            c_cluster = (c_mask[:, :-1] * c_mask[:, 1:])

            v_trigger = torch.zeros_like(v_mask); v_trigger[:, 1:] = v_cluster
            c_trigger = torch.zeros_like(c_mask); c_trigger[:, 1:] = c_cluster

            v_prob = probs[:, :, self.vowel_ids].sum(dim=-1)
            c_prob = probs[:, :, self.consonant_ids].sum(dim=-1)
            phono_penalty = current_phono_lambda * ((v_trigger * v_prob) + (c_trigger * c_prob)).mean()

        # 5. Composite Final Calculation
        current_entropy_lambda = self.config.punishment_scale
        if step < self.config.penalty_warmup_iters:
            current_entropy_lambda *= (step / self.config.penalty_warmup_iters) ** 2

        weighted_norm_entropy = (entropy_per_pos * pos_weights * ctx_weights).mean() / math.log(self.vocab_size)

        return current_entropy_lambda * (weighted_norm_entropy ** 2) + rep_penalty + phono_penalty
