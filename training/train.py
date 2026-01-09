"""
Main entry point for training the GPT model.
"""
import torch
import argparse
import yaml

from .model import GPT, GPTConfig
from .data_loader import DataManager
from .trainer import Trainer

def main(config: dict):
    """
    Orchestrates the training process from configuration.

    Args:
        config (dict): A dictionary containing the training configuration.
    """
    # --- Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Data Manager ---
    data_manager = DataManager(
        data_path=config['data']['path'],
        block_size=config['model']['block_size'],
        batch_size=config['training']['batch_size'],
        device=device
    )

    # --- Model ---
    gpt_config = GPTConfig(
        vocab_size=data_manager.vocab_size,
        block_size=config['model']['block_size'],
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        n_embd=config['model']['n_embd'],
        dropout=config['model']['dropout']
    )
    model = GPT(gpt_config).to(device)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate'])
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_manager=data_manager,
        device=device,
        max_steps=config['training']['max_steps'],
        eval_interval=config['training']['eval_interval'],
        output_dir=config['training']['output_dir']
    )
    trainer.train()
