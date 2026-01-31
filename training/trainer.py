# [INJECTOR: THE PHILOSOPHY OF A MODULAR TRAINER]
#
# The `Trainer` class is the orchestrator of the entire training process. Its primary
# responsibility is to encapsulate the complexity of the training loop, making the
# top-level script (`train.py`) clean and readable.
#
# Key design principles for this module:
# 1.  Encapsulation: The `Trainer` owns the model, the optimizer, and the data manager.
#     It manages their interactions and lifecycle (e.g., initialization, state saving).
#     This prevents concerns from leaking into other parts of the codebase.
#
# 2.  Configuration-Driven: The entire behavior of the trainer (learning rate,
#     batch size, number of steps, etc.) is determined by a single configuration
#     dictionary. This makes experiments easy to define and reproduce. Hardcoded
#     magic numbers are strictly forbidden.
#
# 3.  Modularity: The main `run` method is decomposed into smaller, well-defined
#     private methods (`_run_step`, `_update_lr`, etc.). This improves readability
#     and makes the code easier to debug and maintain. Each private method has a
#     single, clear responsibility.
#
# 4.  Clarity over Cleverness: The code is written to be easily understood. For example,
#     the logic for separating parameters for weight decay in `_build_optimizer` is
#     verbose but explicit, leaving no ambiguity about which parameters are being
#     decayed.
#
# By adhering to these principles, the `Trainer` becomes a robust and flexible
# component of the MLOps pipeline, capable of being adapted for different models
# and datasets with minimal changes.
"""
This module contains the Trainer class, which encapsulates the core logic for
training the GPT model. It handles the model, optimizer, training loop, and
checkpointing, all driven by a configuration dictionary.
"""
import os
import time
import math
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .data_loader import DataManager
from .model import GPT, GPTConfig

class Trainer:
    """
    Orchestrates the model training process.
    This class encapsulates the training loop, model and optimizer setup,
    and checkpointing. It is designed to be configured via a dictionary.
    Attributes:
        config (Dict[str, Any]): The configuration dictionary.
        data_manager (DataManager): The data manager instance.
        device (str): The computing device ('cuda' or 'cpu').
        model (GPT): The GPT model instance.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
    """

    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """Initializes the Trainer.
        Args:
            config: The configuration dictionary.
            data_manager: The data manager instance.
        """
        self.config = config
        self.data_manager = data_manager
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Initialize model and optimizer
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()

    def _build_model(self) -> nn.Module:
        """Builds the GPT model based on the configuration."""
        model_config = self.config['model']
        gpt_config = GPTConfig(
            vocab_size=self.data_manager.vocab_size,
            block_size=model_config['block_size'],
            n_layer=model_config['n_layer'],
            n_head=model_config['n_head'],
            n_embd=model_config['n_embd'],
            dropout=model_config['dropout']
        )
        return GPT(gpt_config).to(self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Builds the AdamW optimizer with a sophisticated weight decay strategy.
        # [INJECTOR: THE WISDOM OF SELECTIVE WEIGHT DECAY]
        # Reference: Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2017)
        # https://arxiv.org/abs/1711.05101
        #
        # Weight decay (L2 regularization) is a technique to prevent overfitting by
        # penalizing large weights. However, applying it blindly to all parameters
        # can be counter-productive.
        #
        # 1.  Linear Weights: Decayed. These are the primary "features" and we want
        #     them to be as simple as possible to generalize well.
        # 2.  Biases: Not decayed. Biases shift the activation function and don't
        #     contribute to the complexity of the feature mapping in the same way
        #     as weights.
        # 3.  LayerNorm/Embedding: Not decayed. These parameters act as "anchors"
        #     for the data distribution (LayerNorm) or map discrete indices to
        #     continuous space (Embedding). Decaying them can destabilize the
        #     model's internal calibration.
        #
        # This method separates model parameters into two groups: those that will
        # experience weight decay and those that will not. Typically, biases,
        # LayerNorm weights, and Embedding weights are not weight-decayed.
        # This helps prevent overfitting without harming model performance.
        Returns:
            torch.optim.Optimizer: The configured AdamW optimizer.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        # Iterate over all named modules and their parameters
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                # Biases are never decayed
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                # Weights of linear layers are decayed
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                # Weights of LayerNorm and Embedding are not decayed
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Sanity checks to ensure every parameter is in one of the sets
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "Parameters in both decay/no_decay sets"
        assert len(param_dict.keys() - union_params) == 0, "Parameters not in decay/no_decay sets"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config['training']['weight_decay']},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        learning_rate = self.config['training']['learning_rate']
        beta1 = self.config['training']['beta1']
        beta2 = self.config['training']['beta2']
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2))

        return optimizer

    def _get_lr(self, it: int) -> float:
        """
        Calculates the learning rate for a given iteration using a cosine decay
        schedule with a linear warmup.
        # [INJECTOR: THE GEOMETRY OF LEARNING RATES]
        # Reference: SGDR: Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter, 2016)
        # https://arxiv.org/abs/1608.03983
        #
        # A static learning rate is rarely optimal. We use a Cosine Decay with Warmup:
        #
        # 1.  Linear Warmup: Gradually increases the LR from 0 to `max_lr`. This
        #     prevents the model from "exploding" early in training when gradients
        #     are noisy and weights are randomly initialized.
        # 2.  Cosine Decay: Follows the curve of a cosine function down to `min_lr`.
        #     This allows for large steps in the beginning (exploration) and
        #     progressively smaller steps at the end (fine-tuning/exploitation),
        #     which is empirically shown to produce better local minima.
        #
        # Args:
            it (int): The current training iteration.
        Returns:
            float: The calculated learning rate.
        """
        learning_rate = self.config['training']['learning_rate']
        min_lr = self.config['training']['min_lr']
        warmup_iters = self.config['training']['warmup_iters']
        lr_decay_iters = self.config['training']['lr_decay_iters']

        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr

        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    def _update_lr(self, step: int) -> Optional[float]:
        """
        Calculates and applies the learning rate for the current training step
        based on a cosine decay schedule with warmup.
        Args:
            step: The current training step.
        Returns:
            The newly calculated learning rate, or None if LR decay is disabled.
        """
        lr = self._get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _run_step(self, scaler: torch.cuda.amp.GradScaler) -> torch.Tensor:
        """
        Executes a single forward and backward pass for one batch of data,
        including gradient scaling and clipping.
        Args:
            scaler: The gradient scaler for mixed-precision training.
        Returns:
            The loss tensor for the current step.
        """
        grad_clip = self.config['training'].get('grad_clip', 1.0)
        xb, yb = self.data_manager.get_batch()

        with torch.amp.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.device == 'cuda')):
            # [INJECTOR: UNPACKING THE TRINITY]
            # The model now returns (logits, loss, presents). During training, we
            # ignore the `presents` (KV-cache) as we are processing full sequences
            # and don't need to store state between steps.
            _, loss, _ = self.model(xb, yb)

        self.optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        scaler.step(self.optimizer)
        scaler.update()

        return loss

    def _log_progress(self, step: int, loss: torch.Tensor, lr: Optional[float]):
        """
        Logs the training progress to the console.
        Args:
            step: The current training step.
            loss: The loss tensor for the current step.
            lr: The learning rate for the current step.
        """
        max_steps = self.config['training']['max_steps']
        if lr is not None:
            print(f"Step {step:4d}/{max_steps}: Loss: {loss.item():.4f}, LR: {lr:.6f}")
        else:
            print(f"Step {step:4d}/{max_steps}: Loss: {loss.item():.4f}")

    def run(self) -> None:
        """
        Executes the main training loop.
        This method orchestrates the training process, including learning rate
        scheduling, executing training steps, and logging progress.
        """
        print("\nStarting training...")
        start_time = time.time()
        max_steps = self.config['training']['max_steps']
        eval_interval = self.config['training']['eval_interval']
        decay_lr = self.config['training'].get('decay_lr', False)

        scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

        for step in range(max_steps):
            lr = self._update_lr(step) if decay_lr else None
            loss = self._run_step(scaler)

            if step % eval_interval == 0 or step == max_steps - 1:
                self._log_progress(step, loss, lr)

        end_time = time.time()
        duration = end_time - start_time
        print(f"Training finished in {duration:.2f} seconds.")
        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        """Saves the model's state dictionary to a checkpoint file."""
        output_dir = self.config['training']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")
