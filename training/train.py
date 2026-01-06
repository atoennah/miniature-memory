import torch
from .data_loader import DataManager
from .trainer import Trainer

def main(config):
    """
    Main function to orchestrate the training process.
    This function initializes the data manager and the trainer, then
    starts the training loop.
    Args:
        config (dict): A dictionary containing the full training configuration.
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
    trainer = Trainer(
        config=config,
        vocab_size=data_manager.vocab_size,
        device=device
    )

    # --- Run Training ---
    trainer.train(data_manager)
