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
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from .data_loader import DataManager
from .model import GPT, GPTConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates the model training process.
    This class encapsulates the training loop, model and optimizer setup,
    and checkpointing. It is designed to be configured via a dictionary,
    making it a flexible component for MLOps pipelines.
    Attributes:
        config (Dict[str, Any]): The configuration dictionary for the trainer.
        data_manager (DataManager): An instance of the data manager.
        device (str): The computing device ('cuda' or 'cpu').
        model (GPT): The GPT model instance.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
    """

    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """Initializes the Trainer.
        This involves setting up the device, building the model, and
        constructing the optimizer.
        Args:
            config (Dict[str, Any]): The configuration dictionary.
            data_manager (DataManager): The data manager instance.
        """
        self.config = config
        self.data_manager = data_manager
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

        self.model = self._build_model()
        self.optimizer = self._build_optimizer()

    def _build_model(self) -> GPT:
        """Builds the GPT model from the configuration.
        This method constructs the GPT model with hyperparameters defined in the
        `model` section of the configuration dictionary.
        Returns:
            GPT: The initialized GPT model, moved to the correct device.
        """
        model_config = self.config['model']
        gpt_config = GPTConfig(
            vocab_size=self.data_manager.vocab_size,
            block_size=model_config['block_size'],
            n_layer=model_config['n_layer'],
            n_head=model_config['n_head'],
            n_embd=model_config['n_embd'],
            dropout=model_config['dropout'],
        )
        return GPT(gpt_config).to(self.device)

    @staticmethod
    def _configure_optimizer_params(
        model: nn.Module,
        weight_decay: float
    ) -> List[Dict[str, Any]]:
        """Separates model parameters into decay and no_decay groups.
        This method identifies parameters that should be subject to weight decay
        (typically weights of linear layers) and those that should not (biases,
        LayerNorm weights, and Embedding weights). This is a standard practice
        in training Transformers.
        Args:
            model (nn.Module): The model for which to configure the optimizer.
            weight_decay (float): The weight decay value for the decay group.
        Returns:
            List[Dict[str, Any]]: A list of parameter groups for the optimizer.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert not inter_params, f"Parameters in both decay/no_decay sets: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, \
            f"Parameters not in decay/no_decay sets: {param_dict.keys() - union_params}"

        return [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Builds the AdamW optimizer with a sophisticated weight decay strategy.
        This method separates model parameters into two groups: those that will
        experience weight decay and those that will not. Typically, biases,
        LayerNorm weights, and Embedding weights are not weight-decayed.
        This helps prevent overfitting without harming model performance.
        Returns:
            torch.optim.Optimizer: The configured AdamW optimizer.
        """
        weight_decay = self.config['training']['weight_decay']
        optim_groups = self._configure_optimizer_params(self.model, weight_decay)

        learning_rate = self.config['training']['learning_rate']
        beta1 = self.config['training']['beta1']
        beta2 = self.config['training']['beta2']
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(beta1, beta2)
        )

        return optimizer

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

        with torch.amp.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.device == 'cuda')):
            _, loss = self.model(xb, yb)

        self.optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        scaler.step(self.optimizer)
        scaler.update()

        return loss

    def _log_progress(self, step: int, loss: float, lr: Optional[float]):
        """Logs the training progress.
        Args:
            step (int): The current training step.
            loss (float): The loss value for the current step.
            lr (Optional[float]): The learning rate for the current step.
        """
        max_steps = self.config['training']['max_steps']
        log_msg = f"Step {step:4d}/{max_steps} | Loss: {loss:.4f}"
        if lr is not None:
            log_msg += f" | LR: {lr:.6f}"
        logger.info(log_msg)

    def run(self) -> None:
        """Executes the main training loop.
        This method orchestrates the training process, including learning rate
        scheduling, executing training steps, and logging progress. It also
        handles mixed-precision training via a gradient scaler.
        """
        logger.info("Starting training...")
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
        logger.info(f"Training finished in {duration:.2f} seconds.")
        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        """Saves the model's state dictionary to a checkpoint file."""
        output_dir = self.config['training']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info(f"Model checkpoint saved to: {checkpoint_path}")
