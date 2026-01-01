import os
import torch
import argparse
import yaml
import time
from model import GPT, GPTConfig

# --- Data Loading and Tokenization ---
def get_data(data_path):
    """Reads the training data and creates a simple char-level tokenizer."""
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    return data, vocab_size, encode, decode

def get_batch(data, block_size, batch_size, device):
    """Generates a small batch of data of inputs x and targets y."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --- Main Training Loop ---
def main(config):
    # --- Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(config['training']['output_dir'], exist_ok=True)

    # --- Data ---
    data, vocab_size, _, _ = get_data(config['data']['path'])

    # --- Model ---
    gpt_config = GPTConfig(
        vocab_size=vocab_size,
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

    # Add some hardcoded values not in the yaml for this simple loop
    config.setdefault('training', {})['max_steps'] = 100
    config.setdefault('training', {})['eval_interval'] = 10
    config.setdefault('training', {})['output_dir'] = 'training/checkpoints'
    config.setdefault('data', {})['path'] = 'dataset/processed/train.txt'

    main(config)
