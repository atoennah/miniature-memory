import argparse
import yaml
import torch

from .data_loader import DataManager
from .trainer import Trainer

def run_training(config):
    """
    Orchestrates the training process by initializing the data manager and trainer.
    """
    # Determine the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the data manager
    data_manager = DataManager(
        data_path=config['data']['path'],
        block_size=config['model']['block_size'],
        batch_size=config['training']['batch_size'],
        device=device
    )

    # Initialize the trainer
    trainer = Trainer(config, data_manager, device)

    # Run the training
    trainer.train()

def main():
    """
    Main entry point for running the training script.
    """
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

    # Set default values for the training configuration
    config.setdefault('training', {})
    config['training'].setdefault('max_steps', 100)
    config['training'].setdefault('eval_interval', 10)
    config['training'].setdefault('output_dir', 'training/checkpoints')
    config['training'].setdefault('learning_rate', 1e-3)
    config['training'].setdefault('batch_size', 32)

    config.setdefault('model', {})
    config['model'].setdefault('block_size', 256)
    config['model'].setdefault('n_layer', 6)
    config['model'].setdefault('n_head', 6)
    config['model'].setdefault('n_embd', 384)
    config['model'].setdefault('dropout', 0.2)

    config.setdefault('data', {})
    config['data'].setdefault('path', 'dataset/processed/train.txt')

    run_training(config)

if __name__ == '__main__':
    main()
