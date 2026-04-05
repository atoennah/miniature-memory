"""
Main script to run the GPT model training process.

This script acts as the orchestrator, bringing together the data manager,
model, and trainer to execute the training loop based on a provided
configuration file.
"""
import argparse
import torch
import yaml
import os
from typing import Dict, Any

from .data_loader import DataManager
from .trainer import Trainer, TrainerConfig
from .model import GPTConfig

def run_training(config: Dict[str, Any]) -> None:
    """
    Orchestrates the model training process by parsing configuration into
    Bolt-standard dataclasses and initializing the Trainer.

    Args:
        config: A dictionary containing the raw training configuration from YAML.
    """
    # 1. Prepare Model Config
    model_cfg_dict = config.get('model', config)
    model_config = GPTConfig(
        vocab_size=0, # Will be determined by DataManager
        block_size=model_cfg_dict.get('block_size', 256),
        n_layer=model_cfg_dict.get('n_layer', 6),
        n_head=model_cfg_dict.get('n_head', 6),
        n_embd=model_cfg_dict.get('n_embd', 384),
        dropout=model_cfg_dict.get('dropout', 0.2)
    )

    # 2. Prepare Trainer Config
    train_cfg_dict = config.get('training', config)
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
        punishment_scale=float(train_cfg_dict.get('punishment_scale', 0.0)),
        penalty_warmup_iters=int(train_cfg_dict.get('penalty_warmup_iters', 0)),
        repetition_penalty=float(train_cfg_dict.get('repetition_penalty', 0.0)),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 3. Initialize the Data Manager
    data_path = config.get('data', config).get('path', 'dataset/processed/train.txt')
    data_manager = DataManager(
        data_path=data_path,
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
    parser.add_argument(
        '--max_steps',
        type=int,
        help='Override max_steps in config.'
    )
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
        return

    if args.max_steps is not None:
        if 'training' in config:
            config['training']['max_steps'] = args.max_steps
        else:
            config['max_steps'] = args.max_steps

    run_training(config)


if __name__ == '__main__':
    main()
