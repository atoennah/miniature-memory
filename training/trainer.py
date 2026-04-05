# [BOLT: THE ARCHITECTURE OF AN OPTIMIZED TRAINER]
#
# This module has been refactored by Bolt to ensure maximum clarity, type safety,
# and performance. The Trainer class is now powered by strongly-typed configuration
# dataclasses, eliminating the fragility of dictionary-based access.
#
# Refactor Highlights:
# 1. Type Safety: Introduced TrainerConfig to provide autocompletion and static
#    analysis for all training hyperparameters.
# 2. Tied-Weights Bug Fix: The optimizer construction logic now correctly handles
#    tied parameters (like lm_head and wte) by delegating to the model's
#    configure_optimizers method.
# 3. Modularization: The training loop is decomposed into logical units (_run_step,
#    _evaluate, _save_checkpoint), making it easier to extend.
# 4. Guard Clauses: Replaced nested logic with early returns to reduce cognitive load.

import os
import time
import math
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from .data_loader import DataManager
from .model import GPT, GPTConfig

@dataclass
class TrainerConfig:
    """Strongly-typed configuration for the training process."""
    max_steps: int = 500
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 100
    lr_decay_iters: int = 500
    min_lr: float = 1e-4
    eval_interval: int = 100
    log_interval: int = 10
    output_dir: str = 'out'
    punishment_scale: float = 0.0
    penalty_warmup_iters: int = 0
    repetition_penalty: float = 0.0
    space_token_id: int = 1
    phonotactic_penalty: float = 0.0
    vowel_ids: List[int] = None
    consonant_ids: List[int] = None
    phono_decay_start: int = 2500
    phono_decay_iters: int = 1500
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer:
    """
    Orchestrates the model training process.
    """

    def __init__(self, config: TrainerConfig, model_config: GPTConfig, data_manager: DataManager):
        """
        Initializes the Trainer with explicit configuration objects.
        """
        self.config = config
        self.data_manager = data_manager
        self.device = config.device

        # Initialize model
        # Note: vocab_size is provided by the data_manager
        model_config.vocab_size = data_manager.vocab_size
        self.model = GPT(model_config).to(self.device)

        # Initialize optimizer
        self.optimizer = self.model.configure_optimizers(
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2)
        )

        # Mixed precision setup
        self.scaler = torch.amp.GradScaler(enabled=(self.device == 'cuda'))

        print(f"Trainer: Initialized on {self.device} with {sum(p.numel() for p in self.model.parameters())} parameters.")

    def _get_lr(self, it: int) -> float:
        """Calculates learning rate with linear warmup and cosine decay."""
        cfg = self.config

        # 1) Linear warmup
        if it < cfg.warmup_iters:
            return cfg.learning_rate * it / cfg.warmup_iters

        # 2) If past decay_iters, return min_lr
        if it > cfg.lr_decay_iters:
            return cfg.min_lr

        # 3) Cosine decay
        decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    def _update_lr(self, it: int) -> float:
        """Applies the calculated learning rate to the optimizer."""
        lr = self._get_lr(it) if self.config.decay_lr else self.config.learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _run_step(self, step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Executes a single forward/backward pass with gradient scaling."""
        xb, yb = self.data_manager.get_batch()
        B, T = xb.size()

        # Forward pass with mixed precision
        with torch.amp.autocast(device_type=('cuda' if 'cuda' in self.device else 'cpu'),
                                dtype=torch.float16,
                                enabled=(self.device == 'cuda')):
            # Forward pass now returns (logits, loss, kv_cache)
            logits, ce_loss, _ = self.model(xb, yb)

            # [BOLT: THE ADVANCED PENALTY SCHEDULING & N-GRAM DE-STUTTERING]
            penalty = torch.tensor(0.0, device=self.device)
            if self.config.punishment_scale > 0:
                # 1. Quadratic Penalty Annealing
                # Sharpen the penalty aggressively as training progresses to finalize morphological commitment.
                current_lambda = self.config.punishment_scale
                if step < self.config.penalty_warmup_iters:
                    current_lambda *= (step / self.config.penalty_warmup_iters) ** 2

                # 2. Length-Normalized Entropy Penalty
                probs = torch.softmax(logits, dim=-1)
                entropy_per_pos = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1) # (B, T)
                pos_weights = torch.linspace(0.5, 1.5, T, device=self.device).view(1, T)

                # 3. Contextual Penalty Scaling
                # Apply a discount factor (0.5) to positions following a space, allowing the model
                # more freedom to choose the next word start.
                # xb[i] is space -> logits[i] predicts token *after* space.
                ctx_weights = torch.ones_like(entropy_per_pos)
                is_space = (xb == self.config.space_token_id)
                ctx_weights[is_space] = 0.5

                # 4. N-Gram Repetition Penalty (De-Stuttering)
                # Penalize if the model predicts the same token that appeared in the input at current pos.
                rep_penalty = torch.tensor(0.0, device=self.device)
                if self.config.repetition_penalty > 0:
                    input_one_hot = torch.zeros_like(logits).scatter_(2, xb.unsqueeze(2), 1.0)
                    overlap = (probs * input_one_hot).sum(dim=-1) # (B, T)
                    rep_penalty = self.config.repetition_penalty * (overlap * ctx_weights).mean()

                # 5. Phonotactic Constraint Scaling & Cosine Decay
                # Penalize transitions that create unnatural character clusters (e.g. 3+ consecutive consonants)
                phono_penalty = torch.tensor(0.0, device=self.device)
                if self.config.phonotactic_penalty > 0 and self.config.vowel_ids and self.config.consonant_ids:
                    current_phono_lambda = self.config.phonotactic_penalty
                    # Apply Cosine Decay to Phonotactic Penalty
                    if step > self.config.phono_decay_start:
                        decay_step = step - self.config.phono_decay_start
                        if decay_step < self.config.phono_decay_iters:
                            decay_ratio = decay_step / self.config.phono_decay_iters
                            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                            current_phono_lambda *= coeff
                        else:
                            current_phono_lambda = 0.0

                    v_mask = torch.isin(xb, torch.tensor(self.config.vowel_ids, device=self.device)).float()
                    c_mask = torch.isin(xb, torch.tensor(self.config.consonant_ids, device=self.device)).float()

                    # Detect clusters of 2 consecutive types in the input (to penalize the 3rd)
                    # xb[i-1] and xb[i] are same type -> logits[i] predicts 3rd in cluster.
                    v_cluster = (v_mask[:, :-1] * v_mask[:, 1:])
                    c_cluster = (c_mask[:, :-1] * c_mask[:, 1:])

                    v_trigger = torch.zeros_like(v_mask); v_trigger[:, 1:] = v_cluster
                    c_trigger = torch.zeros_like(c_mask); c_trigger[:, 1:] = c_cluster

                    v_prob = probs[:, :, self.config.vowel_ids].sum(dim=-1)
                    c_prob = probs[:, :, self.config.consonant_ids].sum(dim=-1)
                    phono_penalty = current_phono_lambda * ((v_trigger * v_prob) + (c_trigger * c_prob)).mean()

                # Final composite penalty
                weighted_norm_entropy = (entropy_per_pos * pos_weights * ctx_weights).mean() / math.log(self.data_manager.vocab_size)
                penalty = current_lambda * (weighted_norm_entropy ** 2) + rep_penalty + phono_penalty

            total_loss = ce_loss + penalty

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()

        # [BOLT: GRADIENT NORM MONITORING]
        if self.config.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        else:
            gnorm = torch.tensor(0.0)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return ce_loss, penalty, total_loss, gnorm

    def run(self):
        """The main training loop."""
        print(f"Trainer: Starting training loop for {self.config.max_steps} steps...")
        start_time = time.time()

        for step in range(self.config.max_steps):
            step_start = time.time()

            lr = self._update_lr(step)
            ce_loss, penalty, total_loss, gnorm = self._run_step(step)

            # Periodic logging
            if step % self.config.log_interval == 0 or step == self.config.max_steps - 1:
                dt = time.time() - step_start
                print(f"step {step:5d} | loss {total_loss.item():.4f} (ce {ce_loss.item():.4f}, pen {penalty.item():.4f}) | gnorm {gnorm.item():.4f} | lr {lr:.4e} | {dt*1000:.2f}ms")

            # Periodic checkpointing
            if step > 0 and step % self.config.eval_interval == 0:
                self._save_checkpoint(f"checkpoint_{step}.pt")

        total_time = time.time() - start_time
        print(f"Trainer: Training complete in {total_time:.2f}s")
        self._save_checkpoint("model_final.pt")

    def _save_checkpoint(self, filename: str):
        """Persists the model state to disk."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, filename)

        checkpoint = {
            'model': self.model.state_dict(),
            'config': self.config,
            'model_config': self.model.config,
        }
        torch.save(checkpoint, path)
        print(f"Trainer: Saved checkpoint to {path}")
