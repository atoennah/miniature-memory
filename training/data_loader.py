"""
Handles loading and preparing the training data.
"""
import torch
from typing import Tuple, List, Dict, Callable

class DataManager:
    """Manages the dataset, including tokenization and batch generation."""

    def __init__(self, data_path: str, block_size: int, batch_size: int, device: str):
        """
        Initializes the DataManager.

        Args:
            data_path (str): The path to the training data file.
            block_size (int): The context length for predictions.
            batch_size (int): The number of independent sequences to process in parallel.
            device (str): The device to move tensors to ('cpu' or 'cuda').
        """
        self.data_path = data_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        self.text: str = self._read_data()
        self.chars: List[str] = sorted(list(set(self.text)))
        self.vocab_size: int = len(self.chars)

        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

        self.encode: Callable[[str], List[int]] = lambda s: [self.stoi[c] for c in s]
        self.decode: Callable[[List[int]], str] = lambda l: ''.join([self.itos[i] for i in l])

        self.data: torch.Tensor = torch.tensor(self.encode(self.text), dtype=torch.long)

    def _read_data(self) -> str:
        """Reads the training data from the specified file."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return f.read()

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a small batch of data of inputs x and targets y.

        Returns:
            A tuple containing the input and target tensors.
        """
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
