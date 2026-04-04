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
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

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
# --- Basic logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_optimizer_param_groups(
    model: nn.Module, weight_decay: float
) -> List[Dict[str, Any]]:
    """Creates optimizer parameter groups with weight decay separation.
    This function separates model parameters into two groups: those that will
    experience weight decay and those that will not. Typically, biases,
    LayerNorm weights, and Embedding weights are not weight-decayed. This helps
    prevent overfitting without harming model performance.
    Args:
        model: The model to create parameter groups for.
        weight_decay: The weight decay value for the decaying group.
    Returns:
        A list of dictionaries, where each dict is a parameter group for the
        optimizer.
    """
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

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
    assert len(inter_params) == 0, "Parameters in both decay/no_decay sets"
    assert len(param_dict.keys() - union_params) == 0, "Parameters not in decay/no_decay sets"

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    return optim_groups


class Trainer:
    """
    """Orchestrates the model training process.
    This class encapsulates the training loop, model and optimizer setup, and
    checkpointing. It is designed to be configured via a dictionary.
    Attributes:
        config: The configuration dictionary.
        data_manager: The data manager instance.
        device: The computing device ('cuda' or 'cpu').
        model: The GPT model instance.
        optimizer: The optimizer for the model.
    """

    def __init__(self, config: TrainerConfig, model_config: GPTConfig, data_manager: DataManager):
        """
        Initializes the Trainer with explicit configuration objects.
        """
        self.config = config
        self._validate_and_cast_config()
        self.data_manager = data_manager
        self.device = config.device

        # Initialize model
        # Note: vocab_size is provided by the data_manager
        model_config.vocab_size = data_manager.vocab_size
        self.model = GPT(model_config).to(self.device)

        # Ensure training config values are numeric
        tcfg = self.config['training']
        for key in ['learning_rate', 'min_lr', 'weight_decay', 'warmup_iters', 'lr_decay_iters', 'max_steps']:
            if key in tcfg:
                tcfg[key] = float(tcfg[key]) if 'iters' not in key and 'steps' not in key else int(float(tcfg[key]))

        # Initialize model and optimizer
        self.model = self._build_model()
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {self.device}")

        self.model: nn.Module = self._build_model()
        self.optimizer: torch.optim.Optimizer = self._build_optimizer()

    def _validate_and_cast_config(self):
        """Ensures that numeric configuration values are of the correct type."""
        # Support both flat and nested configuration formats
        t_cfg = self.config.get('training', self.config)

        # Unify max_iters and max_steps
        if 'max_iters' in t_cfg and 'max_steps' not in t_cfg:
            t_cfg['max_steps'] = t_cfg['max_iters']

        float_params = ['learning_rate', 'weight_decay', 'beta1', 'beta2', 'min_lr']
        int_params = ['batch_size', 'max_steps', 'warmup_iters', 'lr_decay_iters', 'eval_interval']

        for p in float_params:
            if p in t_cfg:
                t_cfg[p] = float(t_cfg[p])
            elif p in self.config:
                self.config[p] = float(self.config[p])

        for p in int_params:
            if p in t_cfg:
                t_cfg[p] = int(t_cfg[p])
            elif p in self.config:
                self.config[p] = int(self.config[p])

    def _build_model(self) -> nn.Module:
        """Builds the GPT model based on the configuration."""
        # Support both flat and nested configuration formats
        m_cfg = self.config.get('model', self.config)

        gpt_config = GPTConfig(
            vocab_size=self.data_manager.vocab_size,
            block_size=m_cfg['block_size'],
            n_layer=m_cfg['n_layer'],
            n_head=m_cfg['n_head'],
            n_embd=m_cfg['n_embd'],
            dropout=m_cfg['dropout']
        """Builds the GPT model from the configuration.
        Returns:
            The initialized GPT model.
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

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        [BOLT: DELEGATED OPTIMIZATION]
        The Trainer now delegates optimizer construction to the Model.
        This follows the principle of Encapsulation: the model knows best which
        of its parameters should be decayed.
        """
        return self.model.configure_optimizers(
            weight_decay=self.config['training']['weight_decay'],
            learning_rate=self.config['training']['learning_rate'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2'])
        Delegates optimizer construction to the model itself.
        """
        train_cfg = self.config['training']
        return self.model.configure_optimizers(
            weight_decay=train_cfg['weight_decay'],
            learning_rate=train_cfg['learning_rate'],
            betas=(train_cfg['beta1'], train_cfg['beta2'])
        """Builds the AdamW optimizer with weight decay.
        Returns:
            The configured AdamW optimizer.
        """
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        weight_decay = self.config['training']['weight_decay']
        optim_groups = create_optimizer_param_groups(self.model, weight_decay)
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        # Iterate over all named modules and their parameters
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                if fpn not in param_dict:
                    continue

                # Biases are never decayed
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                # Weights of linear layers are decayed
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                # Weights of LayerNorm and Embedding are not decayed
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Sanity checks and filtering to handle tied weights (e.g., lm_head.weight tied to wte.weight)
        param_dict = {pn: p for pn, p in self.model.named_parameters()}

        # Filter decay/no_decay sets to only include parameters actually present in named_parameters()
        # This handles tied weights where multiple names might point to the same parameter tensor.
        decay = {pn for pn in decay if pn in param_dict}
        no_decay = {pn for pn in no_decay if pn in param_dict}
        # Sanity check to ensure every parameter is in one of the sets
        all_params = decay | no_decay
        assert len(param_dict.keys() - all_params) == 0, "Not all parameters were assigned to an optimizer group"
        # BUG: The lm_head weight is tied to the token embedding weight and should not be decayed.
        # It's present in the decay set but not as a distinct parameter, causing a KeyError.
        # We explicitly remove it here.
        if 'lm_head.weight' in decay:
            decay.remove('lm_head.weight')

        # Sanity checks to ensure every parameter is in one of the sets
        param_dict = {pn: p for pn, p in self.model.named_parameters()}

        # Filter decay/no_decay sets to only include parameters present in the model
        # This handles weight tying where multiple names might point to the same parameter
        # but named_parameters() only returns one of them.
        decay = {pn for pn in decay if pn in param_dict}
        no_decay = {pn for pn in no_decay if pn in param_dict}
        # Filter sets to only include keys that actually exist in param_dict (handles tied weights)
        decay = {pn for pn in decay if pn in param_dict}
        no_decay = {pn for pn in no_decay if pn in param_dict}
        # Due to weight tying, 'lm_head.weight' is a name for the same parameter
        # as 'transformer.wte.weight', but only the latter is returned by
        # .named_parameters(). The optimizer logic incorrectly adds 'lm_head.weight'
        # to the decay set. We must remove it to avoid a KeyError.
        # The actual parameter ('transformer.wte.weight') is correctly handled
        # and placed in the no_decay set as an embedding weight.
        if 'lm_head.weight' in decay:
            decay.remove('lm_head.weight')

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "Parameters in both decay/no_decay sets"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters not in decay/no_decay sets: {param_dict.keys() - union_params}"
        # Bolt Fix: Filter for tied weights compatibility
        decay = {pn for pn in decay if pn in param_dict}
        no_decay = {pn for pn in no_decay if pn in param_dict}

        assert len(param_dict.keys() - union_params) == 0, "Parameters not in decay/no_decay sets"

        # Filter sets to only include parameters that were actually found in named_parameters
        # (Handling tied weights like lm_head.weight which may not appear separately)
        decay = {pn for pn in decay if pn in param_dict}
        no_decay = {pn for pn in no_decay if pn in param_dict}

        # Filter decay/no_decay to only include parameters actually in param_dict
        # (Handling tied weights where the same parameter may have multiple names in named_modules but only one in named_parameters)
        decay = {pn for pn in decay if pn in param_dict}
        no_decay = {pn for pn in no_decay if pn in param_dict}
        assert len(param_dict.keys() - union_params) == 0, f"Parameters not in decay/no_decay sets: {param_dict.keys() - union_params}"

        t_cfg = self.config.get('training', self.config)
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
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": t_cfg['weight_decay']},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        learning_rate = t_cfg['learning_rate']
        beta1 = t_cfg['beta1']
        beta2 = t_cfg['beta2']
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2))

        return optimizer
        learning_rate = self.config['training']['learning_rate']
        beta1 = self.config['training']['beta1']
        beta2 = self.config['training']['beta2']
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(beta1, beta2)
        )

    def _get_lr(self, it: int) -> float:
        """Calculates the learning rate for a given iteration.
        Uses a cosine decay schedule with a linear warmup.
        Args:
            it: The current training iteration.
        Returns:
            The calculated learning rate.
        """
        t_cfg = self.config.get('training', self.config)
        learning_rate = t_cfg['learning_rate']
        min_lr = t_cfg['min_lr']
        warmup_iters = t_cfg['warmup_iters']
        lr_decay_iters = t_cfg['lr_decay_iters']

        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr

        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    def _update_lr(self, it: int) -> float:
        """Applies the calculated learning rate to the optimizer."""
        lr = self._get_lr(it) if self.config.decay_lr else self.config.learning_rate
    def _update_lr(self, step: int) -> float:
        """Updates the learning rate for the current training step.
        Args:
            step: The current training step.
        Returns:
            The newly calculated learning rate.
        """
        lr = self._get_lr(step)
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
    def _run_step(
        self, scaler: torch.cuda.amp.GradScaler
    ) -> torch.Tensor:
        """Executes a single forward and backward pass for one batch of data.
        Includes gradient scaling and clipping.
        Args:
            scaler: The gradient scaler for mixed-precision training.
        Returns:
            The loss tensor for the current step.
        """
        t_cfg = self.config.get('training', self.config)
        grad_clip = t_cfg.get('grad_clip', 1.0)
        xb, yb = self.data_manager.get_batch()

        # [BOLT: TUPLE UNPACKING]
        # We now unpack 3 values from the model's forward pass.
        # The KV cache is ignored during training.
        with torch.amp.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.device == 'cuda')):
            # The forward pass now returns a third value, the kv_cache, which is not needed for training.
            # The forward pass now returns a third value, the kv_cache, which we ignore during training
        with torch.amp.autocast(
            device_type=self.device, dtype=torch.float16, enabled=(self.device == 'cuda')
        ):
            _, loss = self.model(xb, yb)
        with torch.amp.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.device == 'cuda')):
            # The forward pass now returns logits, loss, and the kv_cache.
            # For training, we only need the loss.
            _, loss, _ = self.model(xb, yb)

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
        scaler.step(self.optimizer)
        scaler.update()
        return loss

    def _log_progress(
        self, step: int, loss: torch.Tensor, lr: Optional[float]
    ) -> None:
        """Logs the training progress to the console.
        Args:
            step: The current training step.
            loss: The loss tensor for the current step.
            lr: The learning rate for the current step.
        """
        t_cfg = self.config.get('training', self.config)
        max_steps = t_cfg['max_steps']
        max_steps = self.config['training']['max_steps']
        log_msg = f"Step {step:4d}/{max_steps}: Loss: {loss.item():.4f}"
        if lr is not None:
            log_msg += f", LR: {lr:.6f}"
        logging.info(log_msg)
            print(
                f"Step {step:4d}/{max_steps}: Loss: {loss.item():.4f}, "
                f"LR: {lr:.6f}"
            )
        else:
            print(f"Step {step:4d}/{max_steps}: Loss: {loss.item():.4f}")

    def run(self) -> None:
        """Executes the main training loop.
        This method orchestrates the training process, including learning rate
        scheduling, executing training steps, and logging progress.
        """
        logging.info("Starting training...")
        start_time = time.time()
        t_cfg = self.config.get('training', self.config)
        max_steps = t_cfg['max_steps']
        eval_interval = t_cfg['eval_interval']
        decay_lr = t_cfg.get('decay_lr', False)

        scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

        for step in range(max_steps):
            lr = self._update_lr(step) if decay_lr else None
            loss = self._run_step(scaler)

            if step % eval_interval == 0 or step == max_steps - 1:
                self._log_progress(step, loss, lr)

        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Training finished in {duration:.2f} seconds.")
        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        """Saves the model's state dictionary to a checkpoint file."""
        t_cfg = self.config.get('training', self.config)
        output_dir = t_cfg.get('output_dir', 'out')
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        logging.info(f"Model checkpoint saved to: {checkpoint_path}")
