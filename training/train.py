import argparse
import os
import time
import torch
import yaml
from typing import Dict, Any

from .model import GPT, GPTConfig
from .data_loader import DataManager

class Trainer:
    """
    A class to encapsulate the training loop for the GPT model.
    It handles model initialization, optimization, and the training process.
    """
    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """
        Initializes the Trainer.

        Args:
            config: A dictionary containing the configuration for the trainer.
            data_manager: An instance of DataManager that handles data loading.
        """
        self.config = config
        self.data_manager = data_manager
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Create output directory if it doesn't exist
        os.makedirs(self.config['training']['output_dir'], exist_ok=True)

        # Get vocab_size from the data_manager
        vocab_size = self.data_manager.vocab_size

        # Initialize model
        gpt_config = GPTConfig(
            vocab_size=vocab_size,
            block_size=self.config['model']['block_size'],
            n_layer=self.config['model']['n_layer'],
            n_head=self.config['model']['n_head'],
            n_embd=self.config['model']['n_embd'],
            dropout=self.config['model']['dropout']
        )
        self.model = GPT(gpt_config).to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate'])
        )

    def _run_step(self, xb: torch.Tensor, yb: torch.Tensor, step: int):
        """Performs a single training step."""
        logits, loss = self.model(xb, yb)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if step % self.config['training']['eval_interval'] == 0:
            print(f"Step {step:4d}/{self.config['training']['max_steps']}: Loss: {loss.item():.4f}")

    def _save_checkpoint(self):
        """Saves the model checkpoint."""
        checkpoint_path = os.path.join(self.config['training']['output_dir'], 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")

    def run(self):
        """Runs the main training loop."""
        print("\nStarting training...")
        start_time = time.time()

        for step in range(self.config['training']['max_steps']):
            xb, yb = self.data_manager.get_batch()
            self._run_step(xb, yb, step)

        end_time = time.time()
        duration = end_time - start_time
        print(f"Training finished in {duration:.2f} seconds.")

        self._save_checkpoint()

def run_training(config: Dict[str, Any]):
    """
    Main function to orchestrate the training pipeline.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
    """
    # Determine the computing device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the DataManager
    data_manager = DataManager(
        data_path=config['data']['path'],
        block_size=config['model']['block_size'],
        batch_size=config['training']['batch_size'],
        device=device
    )

    # Initialize the Trainer
    trainer = Trainer(config, data_manager)

    # Run the training
    trainer.run()

def main():
    """Main entry point for standalone script execution."""
    parser = argparse.ArgumentParser(description="Train a miniature-memory GPT model.")
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
        exit(1)

    run_training(config)

if __name__ == '__main__':
    main()
