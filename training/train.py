"""
This script serves as the main entry point for the training pipeline.

It orchestrates the data loading, model initialization, and the training
process by leveraging the DataManager and Trainer classes.
"""

import argparse
import yaml
import torch
from .model import GPT, GPTConfig
from .data_loader import DataManager
from .trainer import Trainer

def run_training(config: dict):
    """
    Orchestrates the model training process.

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
    trainer = Trainer(model, optimizer, config, device)
    trainer.train(data_manager)

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
        config_from_file = yaml.safe_load(f)

    # Define defaults for standalone execution, which can be overridden by the config file.
    # This maintains compatibility with the main `run.py` orchestrator.
    config = {
        'training': {
            'max_steps': 100,
            'eval_interval': 10,
            'output_dir': 'training/checkpoints',
        },
        'data': {
            'path': 'dataset/processed/train.txt'
        }
    }
    # Deep merge the config from file into the defaults
    for key, value in config_from_file.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value

    run_training(config)
