import os
import torch
from .model import GPT, GPTConfig

class Trainer:
    """
    Manages the model, optimizer, and the training loop.
    This class encapsulates the core training mechanics, allowing the main
    script to be a simple orchestrator.
    """
    def __init__(self, config, vocab_size, device):
        """
        Initializes the Trainer.
        Args:
            config (dict): A dictionary containing the model and training configuration.
            vocab_size (int): The size of the vocabulary.
            device (str): The device to train on ('cpu' or 'cuda').
        """
        self.config = config
        self.device = device
        self.model = self._create_model(vocab_size)
        self.optimizer = self._create_optimizer()

    def _create_model(self, vocab_size):
        """Creates the GPT model based on the configuration."""
        model_config = self.config['model']
        gpt_config = GPTConfig(
            vocab_size=vocab_size,
            block_size=model_config['block_size'],
            n_layer=model_config['n_layer'],
            n_head=model_config['n_head'],
            n_embd=model_config['n_embd'],
            dropout=model_config['dropout']
        )
        return GPT(gpt_config).to(self.device)

    def _create_optimizer(self):
        """Creates the AdamW optimizer for the model."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate'])
        )

    def train(self, data_manager):
        """
        Runs the main training loop.
        Args:
            data_manager (DataManager): The data manager instance for fetching batches.
        """
        print("\nStarting training...")
        for step in range(self.config['training']['max_steps']):
            xb, yb = data_manager.get_batch()

            # Evaluate the loss
            logits, loss = self.model(xb, yb)
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
