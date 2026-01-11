import os
import torch
import yaml
import argparse
from typing import Dict, Any

from .model import GPT, GPTConfig
from .data_loader import DataManager

class Trainer:
    """Manages the model, optimizer, and the core training loop."""

    def __init__(self, config: Dict[str, Any], model: GPT, optimizer: torch.optim.Optimizer, data_manager: DataManager):
        """
        Initializes the Trainer.

        Args:
            config (Dict[str, Any]): The training configuration.
            model (GPT): The GPT model to train.
            optimizer (torch.optim.Optimizer): The optimizer.
            data_manager (DataManager): The data manager.
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.data_manager = data_manager

    def train(self):
        """Runs the main training loop."""
        print("\nStarting training...")
        for step in range(self.config['training']['max_steps']):
            xb, yb = self.data_manager.get_batch()

            _, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if step % self.config['training']['eval_interval'] == 0:
                print(f"Step {step:4d}/{self.config['training']['max_steps']}: Loss: {loss.item():.4f}")

        print("Training finished.")
        self._save_checkpoint()

    def _save_checkpoint(self):
        """Saves the model checkpoint."""
        output_dir = self.config['training']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")

def run_training(config: Dict[str, Any]):
    """
    Sets up and runs the training process.

    Args:
        config (Dict[str, Any]): The training configuration.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Data ---
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

    # --- Training ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate'])
    )

    trainer = Trainer(config, model, optimizer, data_manager)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a NanoGPT model.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/small.yaml',
        help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()

    # Construct the full path to the config file
    # This assumes the script is run from the root of the repository
    config_path = os.path.join('training', args.config)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Hardcoded values for standalone execution
    config.setdefault('training', {})
    config['training'].setdefault('max_steps', 100)
    config['training'].setdefault('eval_interval', 10)
    config['training'].setdefault('output_dir', 'training/checkpoints')

    config.setdefault('data', {})
    config['data'].setdefault('path', 'dataset/processed/train.txt')

    run_training(config)
