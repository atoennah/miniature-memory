"""
Main script to run the GPT model training process.

This script acts as the orchestrator, bringing together the data manager,
model, and trainer to execute the training loop based on a provided
configuration file. It can be called from another script (like `run.py`)
or executed standalone.
"""
import argparse
import os
import yaml
import torch
import torch.nn as nn
from typing import Dict, Any

from .data_loader import DataManager
from .model import GPT, GPTConfig
from .trainer import Trainer


def _build_model(config: Dict[str, Any], data_manager: DataManager) -> nn.Module:
    """Builds the GPT model based on the configuration."""
    model_config = config['model']
    gpt_config = GPTConfig(
        vocab_size=data_manager.vocab_size,
        block_size=model_config['block_size'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        n_embd=model_config['n_embd'],
        dropout=model_config['dropout']
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return GPT(gpt_config).to(device)




def run_training(config: Dict[str, Any]) -> None:
    """
    Orchestrates the model training process.
    This function initializes the data manager, model, and optimizer,
    then starts the training loop.
    Args:
        config: A dictionary containing the training configuration.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_manager = DataManager(
        data_path=config['data']['path'],
        block_size=config['model']['block_size'],
        batch_size=config['training']['batch_size'],
        device=device
    )

    model = _build_model(config, data_manager)
    optimizer = model.configure_optimizers(
        weight_decay=config['training']['weight_decay'],
        learning_rate=config['training']['learning_rate'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )

    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        data_manager=data_manager
    )
    trainer.run()


def main() -> None:
    """
    Main entry point for standalone execution of the training script.

    Parses command-line arguments to load a configuration file and
    starts the training process.
    """
    parser = argparse.ArgumentParser(description="Train a NanoGPT model.")
    parser.add_argument(
        '--config',
        type=str,
        default='training/configs/small.yaml',
        help='Path to the YAML configuration file from the repository root.'
    )
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
        return

    # Provide sensible defaults for standalone execution.
    config.setdefault('training', {}).setdefault('output_dir', 'training/checkpoints')
    config.setdefault('data', {}).setdefault('path', 'dataset/processed/train.txt')

    run_training(config)


if __name__ == '__main__':
    main()
