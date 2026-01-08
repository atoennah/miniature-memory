"""
Handles the training loop for the GPT model.
"""
import os
import torch

class Trainer:
    """Manages the model, optimizer, and the training loop."""

    def __init__(self, config, model, optimizer, data_manager):
        """
        Initializes the Trainer.

        Args:
            config: The training configuration.
            model: The GPT model instance.
            optimizer: The optimizer instance.
            data_manager: The DataManager instance.
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.data_manager = data_manager

    def train(self):
        """Runs the main training loop."""
        print("\nStarting training...")
        for step in range(self.config['training']['max_steps']):
            # Get a batch of data
            xb, yb = self.data_manager.get_batch()

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
