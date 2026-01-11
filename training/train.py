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
from typing import Dict, Any

from .data_loader import DataManager
from .trainer import Trainer

def run_training(config: Dict[str, Any]) -> None:
    """
    Orchestrates the model training process.

    This function initializes the data manager and the trainer, then
    starts the training loop.

    Args:
        config: A dictionary containing the training configuration.
    """
    # Initialize the data manager
    data_manager = DataManager(
        data_path=config['data']['path'],
        block_size=config['model']['block_size'],
        batch_size=config['training']['batch_size'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Initialize and run the trainer
    trainer = Trainer(config=config, data_manager=data_manager)
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
