import torch
from typing import Tuple, List, Dict

class DataManager:
    """Manages data loading, tokenization, and batching for training.

    This class encapsulates the logic for reading the dataset, creating a
    character-level tokenizer, and generating batches for the training loop.

    Attributes:
        data (torch.Tensor): The entire dataset as a single tensor of token IDs.
        vocab_size (int): The number of unique characters in the vocabulary.
        encode (callable): A function to encode a string into a list of token IDs.
        decode (callable): A function to decode a list of token IDs into a string.
        block_size (int): The context length for the model.
        batch_size (int): The number of sequences in a batch.
        device (str): The device to move tensors to ('cpu' or 'cuda').
    """

    def __init__(self, data_path: str, block_size: int, batch_size: int, device: str):
        """Initializes the DataManager and loads the data.

        Args:
            data_path (str): The path to the training data file (e.g., 'train.txt').
            block_size (int): The context length for the transformer model.
            batch_size (int): The number of independent sequences in a batch.
            device (str): The computing device ('cuda' or 'cpu').
        """
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self._load_data(data_path)

    def _load_data(self, data_path: str) -> None:
        """Reads data, creates a tokenizer, and prepares the dataset tensor."""
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: ''.join([itos[i] for i in l])

        self.data = torch.tensor(self.encode(text), dtype=torch.long)
        print(f"Loaded {len(self.data)} tokens from '{data_path}'")
        print(f"Vocabulary size: {self.vocab_size}")

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates a small batch of data of inputs x and targets y.

        Returns:
            A tuple containing:
            - A (batch_size, block_size) tensor of input token IDs.
            - A (batch_size, block_size) tensor of target token IDs.
        """
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
