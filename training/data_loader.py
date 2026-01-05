import torch
from typing import Tuple, List, Dict, Callable

class DataManager:
    """Manages loading and batching of training data."""

    def __init__(self, data_path: str, block_size: int, batch_size: int, device: str):
        """
        Initializes the DataManager.

        Args:
            data_path (str): Path to the training data file.
            block_size (int): The context length for predictions.
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
        Reads the training data and creates a simple char-level tokenizer.

        Returns:
            Tuple[torch.Tensor, int, Callable[[str], List[int]], Callable[[List[int]], str]]:
                - The encoded training data as a tensor.
                - The vocabulary size.
                - A function to encode a string to a list of integers.
                - A function to decode a list of integers to a string.
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}

        encode: Callable[[str], List[int]] = lambda s: [stoi[c] for c in s]
        decode: Callable[[List[int]], str] = lambda l: ''.join([itos[i] for i in l])

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
