"""
Main script to run the GPT model training process.

This script acts as the orchestrator, bringing together the data manager,
model, and trainer to execute the training loop based on a provided
configuration file. It can be called from another script (like `run.py`)
or executed standalone.
"""
import argparse
import torch
from typing import Dict, Any

from .data_loader import DataManager
from .trainer import Trainer
from .model import GPTConfig

def run_training(config_path: str) -> None:
    """
    Orchestrates the model training process.

    This function initializes the data manager and the trainer, then
    starts the training loop.

    Args:
        config_path: Path to the YAML configuration file.
    """
    # Load the configuration using the robust `from_yaml` method
    config = GPTConfig.from_yaml(config_path)

    # Pass the full config dictionary to legacy components for now.
    # TODO: Refactor DataManager and Trainer to accept the GPTConfig object directly.
    import yaml
    with open(config_path, 'r') as f:
        legacy_config = yaml.safe_load(f)

    # Initialize the data manager
    data_manager = DataManager(
        data_path=legacy_config['data']['path'],
        block_size=config.block_size,
        batch_size=legacy_config['training']['batch_size'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Initialize and run the trainer
    trainer = Trainer(config=legacy_config, data_manager=data_manager)
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

    run_training(args.config)


if __name__ == '__main__':
    main()
