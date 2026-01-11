import argparse
import torch
import yaml
from typing import Dict, Any
import time
from model import GPT, GPTConfig

# --- Data Loading and Tokenization ---
def get_data(data_path):
    """Reads the training data and creates a simple char-level tokenizer."""
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

from .data_loader import DataManager
from .trainer import Trainer

def main(config: Dict[str, Any]) -> None:
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
    # --- Training ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']) # Explicitly cast to float
    )

    print("\nStarting training...")
    start_time = time.time()
    for step in range(config['training']['max_steps']):
        # Get a batch of data
        xb, yb = get_batch(
            data,
            gpt_config.block_size,
            config['training']['batch_size'],
            device
        )

        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % config['training']['eval_interval'] == 0:
            print(f"Step {step:4d}/{config['training']['max_steps']}: Loss: {loss.item():.4f}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Training finished in {duration:.2f} seconds.")

    # --- Save Checkpoint ---
    checkpoint_path = os.path.join(config['training']['output_dir'], 'model.pt')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModel checkpoint saved to: {checkpoint_path}")

    # Run the training
    trainer.run()

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

    # --- Configuration Defaults ---
    # These values are set to ensure that even a minimal config file will run,
    # especially for standalone execution of this script.
    config.setdefault('training', {})
    config['training'].setdefault('max_steps', 100)
    config['training'].setdefault('eval_interval', 10)
    config['training'].setdefault('output_dir', 'training/checkpoints')

    config.setdefault('data', {})
    config['data'].setdefault('path', 'dataset/processed/train.txt')

    main(config)
