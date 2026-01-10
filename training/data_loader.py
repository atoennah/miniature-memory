"""
Handles loading and preparing the dataset for training.
"""

import torch

class DataManager:
    """
    Manages the training data, including loading, tokenization, and batching.
    """
    def __init__(self, data_path: str, block_size: int, batch_size: int, device: str):
        """
        Initializes the DataManager.

        Args:
            data_path (str): The path to the training data file.
            block_size (int): The context size for the model.
            batch_size (int): The number of sequences in a batch.
            device (str): The device to move tensors to ('cpu' or 'cuda').
        """
        self.data_path = data_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        self.text = self._read_data()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        self._stoi = {ch: i for i, ch in enumerate(self.chars)}
        self._itos = {i: ch for i, ch in enumerate(self.chars)}

        self.data = self._encode_text()

    def _read_data(self) -> str:
        """Reads the training data from the specified path."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _encode_text(self) -> torch.Tensor:
        """Encodes the text into a tensor of integers."""
        encoded_text = [self._stoi[c] for c in self.text]
        return torch.tensor(encoded_text, dtype=torch.long)

    def encode(self, s: str) -> list[int]:
        """Encodes a string using the vocabulary."""
        return [self._stoi[c] for c in s]

    def decode(self, l: list[int]) -> str:
        """Decodes a list of integers back to a string."""
        return ''.join([self._itos[i] for i in l])

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
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
