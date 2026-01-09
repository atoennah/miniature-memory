import os
import torch
from .model import GPT, GPTConfig

class Trainer:
    """Manages the model training loop."""
    def __init__(self, config, data_manager, device):
        self.config = config
        self.data_manager = data_manager
        self.device = device
        print(f"Using device: {self.device}")

        # Create output directory
        os.makedirs(self.config['training']['output_dir'], exist_ok=True)

        # Build the model
        gpt_config = GPTConfig(
            vocab_size=self.data_manager.vocab_size,
            block_size=self.config['model']['block_size'],
            n_layer=self.config['model']['n_layer'],
            n_head=self.config['model']['n_head'],
            n_embd=self.config['model']['n_embd'],
            dropout=self.config['model']['dropout']
        )
        self.model = GPT(gpt_config).to(self.device)

        # Create the optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate'])
        )

    def train(self):
        """Runs the training loop for the specified number of steps."""
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
        checkpoint_path = os.path.join(self.config['training']['output_dir'], 'model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")
