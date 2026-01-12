"""
This module contains the Trainer class, which encapsulates the core logic
for training the GPT model. It handles model and optimizer initialization,
the main training loop, validation, and checkpointing.
"""
import os
import time
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from typing import Dict, Any, Tuple

from .data_loader import DataManager
from .model import GPT, GPTConfig

class Trainer:
    """
    Orchestrates an efficient and well-documented model training process.

    This class encapsulates the training loop, model and optimizer setup,
    learning rate scheduling, validation, and checkpointing. It is designed
    to be configured via a dictionary and prioritizes clarity and best
    practices like gradient clipping and mixed-precision training.

    Attributes:
        config (Dict[str, Any]): The configuration dictionary.
        data_manager (DataManager): The data manager for training and validation data.
        device (str): The computing device ('cuda', 'mps', or 'cpu').
        model (GPT): The GPT model instance.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        scheduler (LRScheduler): The learning rate scheduler.
        scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed-precision.
    """

    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """
        Initializes the Trainer by setting up the device, model, optimizer,
        and learning rate scheduler.
        """
        self.config = config
        self.data_manager = data_manager
        self.device = self._setup_device()

        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_lr_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

    def _setup_device(self) -> str:
        """
        ⚡ **Bolt Feature:** Determines the best available compute device.
        Prioritizes CUDA, then Apple's MPS, and falls back to CPU.
        """
        if torch.cuda.is_available():
            print("Using device: CUDA")
            return 'cuda'
        elif torch.backends.mps.is_available():
            print("Using device: MPS")
            return 'mps'
        else:
            print("Using device: CPU")
            return 'cpu'

    def _build_model(self) -> nn.Module:
        """Builds the GPT model from the configuration."""
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
        """Builds the AdamW optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate'])
        )

    def _get_lr_scheduler(self) -> LRScheduler:
        """
        ⚡ **Bolt Feature:** Implements a cosine learning rate scheduler.
        This helps the model converge more effectively by gradually
        decreasing the learning rate.
        """
        training_config = self.config['training']
        max_steps = training_config['max_steps']
        warmup_steps = training_config.get('warmup_steps', 0)
        min_lr = float(training_config.get('min_lr', 1e-5))

        def lr_lambda(it):
            if it < warmup_steps:
                return float(it) / float(max(1, warmup_steps))
            if it > max_steps:
                return min_lr

            decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
            assert 0 <= decay_ratio <= 1

            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (training_config['learning_rate'] - min_lr)

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    @torch.no_grad()
    def _validate_model(self) -> Dict[str, float]:
        """
        ⚡ **Bolt Feature:** Runs a model validation loop.
        Calculates and returns the average loss over the validation set.
        """
        self.model.eval()
        losses = torch.zeros(self.config['training']['eval_iters'])
        for k in range(self.config['training']['eval_iters']):
            X, Y = self.data_manager.get_batch('val')
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.device == 'cuda')):
                _, loss = self.model(X, Y)
            losses[k] = loss.item()
        self.model.train()
        return {'val_loss': losses.mean().item()}

    def run(self) -> None:
        """
        Executes the main training loop, including validation, logging,
        and learning rate scheduling.
        """
        print("\n⚡ Starting training with Bolt optimizations...")
        start_time = time.time()

        training_config = self.config['training']
        max_steps = training_config['max_steps']
        eval_interval = training_config['eval_interval']
        grad_clip = training_config.get('grad_clip', 1.0)

        for step in range(max_steps):
            t0 = time.time()

            xb, yb = self.data_manager.get_batch('train')

            with torch.amp.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.device == 'cuda')):
                logits, loss = self.model(xb, yb)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            if grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()

            if step % eval_interval == 0 or step == max_steps - 1:
                metrics = self._validate_model()
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"Step {step:4d}/{max_steps}: "
                    f"Train Loss: {loss.item():.4f}, "
                    f"Val Loss: {metrics['val_loss']:.4f}, "
                    f"LR: {lr:.6f}, "
                    f"Time: {(time.time() - t0) * 1000:.2f}ms"
                )

        duration = time.time() - start_time
        print(f"✓ Training finished in {duration:.2f} seconds.")
        self._save_checkpoint(step)

    def _save_checkpoint(self, step: int) -> None:
        """
        ⚡ **Bolt Feature:** Saves a comprehensive checkpoint.
        Includes model state, optimizer state, scheduler state, and config,
        allowing for seamless resumption of training.
        """
        output_dir = self.config['training']['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'step': step,
        }

        checkpoint_path = os.path.join(output_dir, f'ckpt_step_{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")
