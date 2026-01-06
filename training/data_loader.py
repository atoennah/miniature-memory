import torch
from typing import Tuple, List, Callable

class DataManager:
    """Manages loading and batching of training data."""

    def __init__(self, data_path: str, block_size: int, batch_size: int, device: str):
        """
        Initializes the DataManager.

        Args:
            data_path (str): Path to the training data file.
            block_size (int): The context size for predictions.
            batch_size (int): The number of independent sequences to process in parallel.
            device (str): The device to move tensors to ('cpu' or 'cuda').
        """
        self.data_path = data_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.data, self.vocab_size, self.encode, self.decode = self._load_data()

    def _load_data(self) -> Tuple[torch.Tensor, int, Callable[[str], List[int]], Callable[[List[int]], str]]:
        """
        Reads the training data, creates a char-level tokenizer, and returns the encoded data.

        Returns:
            Tuple[torch.Tensor, int, Callable, Callable]: A tuple containing the encoded data tensor,
                                                          vocabulary size, encoder function, and decoder function.
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])

        data = torch.tensor(encode(text), dtype=torch.long)
        return data, vocab_size, encode, decode

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a small batch of data of inputs x and targets y.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target tensors.
        """
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
