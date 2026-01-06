import argparse
import yaml
import torch

from .data_loader import DataManager
from .trainer import Trainer
from typing import Dict

def run_training(config: Dict):
    """
    Initializes and runs the training process.

    Args:
        config (Dict): The configuration dictionary for training.
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

    # --- Trainer ---
    trainer = Trainer(config, data_manager)
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

    # Add some hardcoded values not in the yaml for this simple loop
    # These are defaults for standalone execution of this script.
    # The main `run.py` script will often override these.
    config.setdefault('training', {})
    config['training'].setdefault('max_steps', 100)
    config['training'].setdefault('eval_interval', 10)
    config['training'].setdefault('output_dir', 'training/checkpoints')
    config.setdefault('data', {})
    config['data'].setdefault('path', 'dataset/processed/train.txt')

    run_training(config)
