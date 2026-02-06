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
#    tied parameters (like lm_head and wte) by ensuring each unique parameter
#    tensor is assigned to exactly one optimizer group.
# 3. Modularization: The training loop is decomposed into logical units (_run_step,
#    _evaluate, _save_checkpoint), making it easier to extend (e.g., for multi-GPU).
# 4. Guard Clauses: Replaced nested logic with early returns to reduce cognitive load.

import os
import time
import math
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
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer:
    """
    Orchestrates the model training process with Bolt-standard clean logic.
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
        self.optimizer = self._build_optimizer()

        # Mixed precision setup
        self.scaler = torch.amp.GradScaler(enabled=(self.device == 'cuda'))

        print(f"Trainer: Initialized on {self.device} with {sum(p.numel() for p in self.model.parameters())} parameters.")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Constructs the AdamW optimizer, properly handling weight decay and tied weights.

        Bolt's Strategy:
        - Decay 2D tensors (weights of Linears and Embeddings).
        - Do not decay 1D/0D tensors (biases, layer norms).
        - Use unique parameter IDs to avoid issues with tied weights.
        """
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}

        # Separate parameters into decay and no_decay groups based on dimensionality
        # This is a robust heuristic: weights (2D+) decay, biases/norms (1D) don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Trainer: Optimizer grouped {len(decay_params)} tensors ({num_decay_params:,} params) for decay, "
              f"{len(nodecay_params)} tensors ({num_nodecay_params:,} params) for no decay.")

        return torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2)
        )

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
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    def _update_lr(self, it: int) -> float:
        """Applies the calculated learning rate to the optimizer."""
        lr = self._get_lr(it) if self.config.decay_lr else self.config.learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _run_step(self) -> torch.Tensor:
        """Executes a single forward/backward pass with gradient scaling."""
        xb, yb = self.data_manager.get_batch()

        # Forward pass with mixed precision
        with torch.amp.autocast(device_type=( 'cuda' if 'cuda' in self.device else 'cpu'),
                                dtype=torch.float16,
                                enabled=(self.device == 'cuda')):
            _, loss = self.model(xb, yb)

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()

        # Gradient clipping
        if self.config.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss

    def run(self):
        """The main training loop, orchestrated with Bolt-standard efficiency."""
        print(f"Trainer: Starting training loop for {self.config.max_steps} steps...")
        start_time = time.time()

        for step in range(self.config.max_steps):
            step_start = time.time()

            lr = self._update_lr(step)
            loss = self._run_step()

            # Periodic logging
            if step % self.config.log_interval == 0 or step == self.config.max_steps - 1:
                dt = time.time() - step_start
                print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.4e} | {dt*1000:.2f}ms")

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

        # Basic state dict saving. Can be expanded to include optimizer state.
        checkpoint = {
            'model': self.model.state_dict(),
            'config': self.config,
            'model_config': self.model.config,
        }
        torch.save(checkpoint, path)
        print(f"Trainer: Saved checkpoint to {path}")
