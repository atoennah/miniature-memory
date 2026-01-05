import argparse
import yaml
from typing import Dict, Any
import torch

from .data_loader import DataManager
from .trainer import Trainer

def main(config: Dict[str, Any]):
    """
    Main function to run the training pipeline.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
    """
    # --- Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Data ---
    data_manager = DataManager(
        data_path=config['data']['path'],
        block_size=config['model']['block_size'],
        batch_size=config['training']['batch_size'],
        device=device
    )

    # --- Training ---
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
    config.setdefault('training', {})['max_steps'] = 100
    config.setdefault('training', {})['eval_interval'] = 10
    config.setdefault('training', {})['output_dir'] = 'training/checkpoints'
    config.setdefault('data', {})['path'] = 'dataset/processed/train.txt'
    # Add block_size to the config if not present, as DataManager needs it.
    config.setdefault('model', {})['block_size'] = 256


    main(config)
