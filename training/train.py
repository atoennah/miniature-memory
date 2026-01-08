"""
Main script to run the GPT model training process.

This script acts as the orchestrator, bringing together the data manager,
model, and trainer to execute the training loop based on a provided
configuration file.
"""
import argparse
import yaml
import torch

from .data_loader import DataManager
from .model import GPT, GPTConfig
from .trainer import Trainer

def run_training(config: dict):
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
    trainer = Trainer(config, model, optimizer, data_manager)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a NanoGPT model.")
    parser.add_argument(
        '--config',
        type=str,
        default='training/configs/small.yaml',
        help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --- Configuration Defaults for Standalone Execution ---
    # These defaults are applied when running the script directly,
    # allowing for quick tests without modifying the main run.py orchestrator.
    config.setdefault('training', {})
    config['training'].setdefault('max_steps', 100)
    config['training'].setdefault('eval_interval', 10)
    config['training'].setdefault('output_dir', 'training/checkpoints')
    config.setdefault('data', {})
    config['data'].setdefault('path', 'dataset/processed/train.txt')

    run_training(config)
