"""
Main script to run the GPT model training process.

This script acts as the orchestrator, bringing together the data manager,
model, and trainer to execute the training loop based on a provided
configuration file.
"""
import argparse
import torch
from typing import Dict, Any

from .data_loader import DataManager
from .trainer import Trainer, TrainerConfig
from .trainer import Trainer
from .model import GPTConfig

def run_training(config_path: str) -> None:
    """
    Orchestrates the model training process by parsing configuration into
    Bolt-standard dataclasses and initializing the Trainer.

    Args:
        config: A dictionary containing the raw training configuration from YAML.
    """
    # 1. Prepare Model Config
    model_cfg_dict = config.get('model', {})
    model_config = GPTConfig(
        vocab_size=None, # Will be determined by DataManager
        block_size=model_cfg_dict.get('block_size', 256),
        n_layer=model_cfg_dict.get('n_layer', 6),
        n_head=model_cfg_dict.get('n_head', 6),
        n_embd=model_cfg_dict.get('n_embd', 384),
        dropout=model_cfg_dict.get('dropout', 0.2)
    )

    # 2. Prepare Trainer Config
    train_cfg_dict = config.get('training', {})
    trainer_config = TrainerConfig(
        max_steps=train_cfg_dict.get('max_steps', 500),
        batch_size=train_cfg_dict.get('batch_size', 32),
        learning_rate=float(train_cfg_dict.get('learning_rate', 1e-3)),
        weight_decay=float(train_cfg_dict.get('weight_decay', 0.1)),
        beta1=float(train_cfg_dict.get('beta1', 0.9)),
        beta2=float(train_cfg_dict.get('beta2', 0.99)),
        grad_clip=float(train_cfg_dict.get('grad_clip', 1.0)),
        decay_lr=train_cfg_dict.get('decay_lr', True),
        warmup_iters=train_cfg_dict.get('warmup_iters', 100),
        lr_decay_iters=train_cfg_dict.get('lr_decay_iters', 500),
        min_lr=float(train_cfg_dict.get('min_lr', 1e-4)),
        eval_interval=train_cfg_dict.get('eval_interval', 100),
        log_interval=train_cfg_dict.get('log_interval', 10),
        output_dir=train_cfg_dict.get('output_dir', 'out'),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 3. Initialize the Data Manager
    data_manager = DataManager(
        data_path=config['data']['path'],
        block_size=model_config.block_size,
        batch_size=trainer_config.batch_size,
        device=trainer_config.device
    )

    # 4. Initialize and run the Trainer
    trainer = Trainer(
        config=trainer_config,
        model_config=model_config,
        data_manager=data_manager
    )
        config_path: Path to the YAML configuration file.
    """
    # Support both flat and nested configuration formats
    data_cfg = config.get('data', config)
    model_cfg = config.get('model', config)
    train_cfg = config.get('training', config)

    # Initialize the data manager
    data_manager = DataManager(
        data_path=data_cfg['path'],
        block_size=model_cfg['block_size'],
        batch_size=train_cfg['batch_size'],
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
    """
    parser = argparse.ArgumentParser(description="Train a NanoGPT model.")
    parser.add_argument(
        '--config',
        type=str,
        default='training/configs/small.yaml',
        help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
        return

    # Sensible defaults for required top-level keys
    config.setdefault('training', {})
    config.setdefault('data', {}).setdefault('path', 'dataset/processed/train.txt')
    config.setdefault('model', {})
    # Provide sensible defaults for standalone execution.
    # We only add nested keys if the config already uses them, otherwise we stay flat.
    if 'training' in config:
        config['training'].setdefault('output_dir', 'training/checkpoints')
    else:
        config.setdefault('output_dir', 'training/checkpoints')

    if 'data' in config:
        config['data'].setdefault('path', 'dataset/processed/train.txt')
    else:
        config.setdefault('path', 'dataset/processed/train.txt')

    run_training(config)
    run_training(args.config)


if __name__ == '__main__':
    main()
