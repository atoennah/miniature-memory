
import time
import torch
import yaml
from typing import Dict, Any

from training.model import GPT, GPTConfig
from training.trainer import Trainer
from training.data_loader import DataManager


def create_mock_data_manager(config: Dict[str, Any], vocab_size: int) -> DataManager:
    """Creates a mock DataManager that yields the same batch repeatedly."""
    class MockDataManager(DataManager):
        def __init__(self, config: Dict[str, Any], vocab_size: int):
            self.batch_size = config['training']['batch_size']
            self.block_size = config['model']['block_size']
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.vocab_size = vocab_size

            # Create a single, fixed batch of dummy data
            self.mock_x = torch.randint(0, self.vocab_size, (self.batch_size, self.block_size), device=self.device)
            self.mock_y = torch.randint(0, self.vocab_size, (self.batch_size, self.block_size), device=self.device)

        def get_batch(self):
            return self.mock_x, self.mock_y

    return MockDataManager(config, vocab_size)


def run_benchmark(config_path: str = 'training/configs/benchmark.yaml') -> float:
    """
    Benchmarks the training step throughput, providing a realistic measure of
    end-to-end training performance.
    This function initializes a model, optimizer, and a mock data manager,
    then uses the Trainer's `_run_step` method to measure performance.
    Args:
        config_path: Path to the benchmark configuration file.
    Returns:
        The measured throughput in tokens per second.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    vocab_size = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Set up mock components
    data_manager = create_mock_data_manager(config, vocab_size)
    model_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=config['model']['block_size'],
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        n_embd=config['model']['n_embd'],
        dropout=config['model']['dropout']
    )
    model = GPT(model_config).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=config['training']['weight_decay'],
        learning_rate=config['training']['learning_rate'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )

    # 2. Instantiate the Trainer
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        data_manager=data_manager
    )

    # 3. Run the benchmark
    num_steps = 20
    warmup_steps = 5
    scaler = trainer.precision_manager.get_scaler()

    # Warmup phase
    for _ in range(warmup_steps):
        trainer._run_step(scaler)

    # Benchmark phase
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    for _ in range(num_steps):
        trainer._run_step(scaler)
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()

    # 4. Calculate and report throughput
    total_time = end_time - start_time
    tokens_per_step = data_manager.batch_size * data_manager.block_size
    total_tokens = num_steps * tokens_per_step
    tokens_per_second = total_tokens / total_time

    print("\n--- Benchmark Results ---")
    print(f"Configuration: {config_path}")
    print(f"Device: {device.upper()}")
    print(f"Steps: {num_steps}")
    print(f"Batch Size: {data_manager.batch_size}")
    print(f"Block Size: {data_manager.block_size}")
    print(f"Total Tokens Processed: {total_tokens:,}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Throughput: {tokens_per_second:,.2f} tokens/sec")
    print("-------------------------\n")

    return tokens_per_second

if __name__ == "__main__":
    run_benchmark()
