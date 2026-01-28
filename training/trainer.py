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
import logging
from typing import Dict, Any, Optional, ContextManager

from .data_loader import DataManager


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PrecisionManager:
    """
    Manages mixed-precision training settings.
    This helper class abstracts away the device-specific logic for mixed-precision
    training, providing a clean interface for using `autocast` and `GradScaler`.
    """
    def __init__(self, device: str):
        self.device = device
        self.dtype = torch.float16 if device == 'cuda' else torch.bfloat16
        self.enabled = (self.device == 'cuda') # bfloat16 CPU is too slow

    def get_autocast(self) -> ContextManager:
        """Returns the autocast context manager for the current device."""
        return torch.amp.autocast(
            device_type=self.device,
            dtype=self.dtype,
            enabled=self.enabled
        )

    def get_scaler(self) -> torch.cuda.amp.GradScaler:
        """Returns a GradScaler for mixed-precision training."""
        return torch.cuda.amp.GradScaler(enabled=self.enabled)


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

    def __init__(self,
                 config: Dict[str, Any],
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 data_manager: DataManager):
        """
        Initializes the Trainer.
        Args:
            config: The configuration dictionary.
            model: The model to be trained.
            optimizer: The optimizer for the model.
            data_manager: The data manager instance.
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.data_manager = data_manager
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.precision_manager = PrecisionManager(self.device)
        logger.info(f"Using device: {self.device}")

    def _get_lr(self, it: int) -> float:
        """
        Calculates the learning rate for a given iteration using a cosine decay
        schedule with a linear warmup.
        Args:
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

        with self.precision_manager.get_autocast():
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
        log_msg = f"Step {step:4d}/{max_steps}: Loss: {loss.item():.4f}"
        if lr is not None:
            log_msg += f", LR: {lr:.6f}"
        logger.info(log_msg)

    def run(self) -> None:
        """
        Executes the main training loop.
        This method orchestrates the training process, including learning rate
        scheduling, executing training steps, and logging progress.
        """
        logger.info("Starting training...")
        start_time = time.time()
        max_steps = self.config['training']['max_steps']
        eval_interval = self.config['training']['eval_interval']
        decay_lr = self.config['training'].get('decay_lr', False)

        scaler = self.precision_manager.get_scaler()

        for step in range(max_steps):
            lr = self._update_lr(step) if decay_lr else None
            loss = self._run_step(scaler)

            if step % eval_interval == 0 or step == max_steps - 1:
                self._log_progress(step, loss, lr)

        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Training finished in {duration:.2f} seconds.")
        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        """Saves the model's state dictionary to a checkpoint file."""
        output_dir = self.config['training']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info(f"Model checkpoint saved to: {checkpoint_path}")
