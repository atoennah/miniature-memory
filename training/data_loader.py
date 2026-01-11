import torch

class DataManager:
    """Manages data loading, tokenization, and batching."""
    def __init__(self, data_path, block_size, batch_size, device):
"""
Handles loading and preparing the training data.
"""
import torch
from typing import Tuple, List, Dict, Callable

class DataManager:
    """Manages the dataset, including tokenization and batch generation."""
Handles loading and preparing the training data for the GPT model.
"""
import torch

class DataManager:
    """Manages the training data, including loading, tokenization, and batching."""
import torch

class DataManager:
    """
    Manages loading, tokenizing, and batching of the training data.
    This class encapsulates all data-related logic to keep the training
    script clean and focused on the training loop itself.
    """
    def __init__(self, data_path, block_size, batch_size, device):
        """
        Initializes the DataManager.
        Args:
            data_path (str): The path to the training data file.
from typing import Tuple, List, Callable
from typing import Tuple, List, Dict, Callable

class DataManager:
    """Manages loading and batching of training data."""

    def __init__(self, data_path: str, block_size: int, batch_size: int, device: str):
        """
        Initializes the DataManager.

        Args:
            data_path (str): The path to the training data file.
            data_path (str): Path to the training data file.
            block_size (int): The context size for predictions.
            block_size (int): The context length for predictions.
            batch_size (int): The number of independent sequences to process in parallel.
            device (str): The device to move tensors to ('cpu' or 'cuda').
        """
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        # Read the raw text
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Build a character-level tokenizer from the text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.data_path = data_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        # Read the text file once.
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()

        self.vocab_size, self.encode, self.decode = self._build_tokenizer()
        self.data = self._load_data()

    def _load_data(self):
        """Encodes the entire text dataset."""
        return torch.tensor(self.encode(self.text), dtype=torch.long)

    def _build_tokenizer(self):
        """Creates a simple char-level tokenizer from the text."""
        chars = sorted(list(set(self.text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        return vocab_size, encode, decode

    def get_batch(self):
        """Generates a small batch of data of inputs x and targets y."""
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

        self.data, self.vocab_size, self.encode, self.decode = self._load_and_tokenize()

    def _load_and_tokenize(self):
        """
        Reads the training data, creates a char-level tokenizer, and encodes the text.

        Returns:
            Tuple[torch.Tensor, int, Callable, Callable]: A tuple containing the encoded
                data tensor, the vocabulary size, and the encoder/decoder functions.
        self.data, self.vocab_size, self.encode, self.decode = self._load_data()

    def _load_data(self) -> Tuple[torch.Tensor, int, Callable[[str], List[int]], Callable[[List[int]], str]]:
        """
        Reads the training data, creates a char-level tokenizer, and returns the encoded data.

        Returns:
            Tuple[torch.Tensor, int, Callable, Callable]: A tuple containing the encoded data tensor,
                                                          vocabulary size, encoder function, and decoder function.
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

        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}

        encode: Callable[[str], List[int]] = lambda s: [stoi[c] for c in s]
        decode: Callable[[List[int]], str] = lambda l: ''.join([itos[i] for i in l])

        data = torch.tensor(encode(text), dtype=torch.long)
        return data, vocab_size, encode, decode

    def get_batch(self):
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a small batch of data of inputs x and targets y.

        Returns:
            A tuple containing the input and target tensors.
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target tensors.
from typing import Tuple, List, Dict
"""
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

        # Encode the entire dataset and store it as a tensor
        self.data = torch.tensor(self.encode(text), dtype=torch.long)

    def get_batch(self):
        """
        Generates a small batch of data of inputs x and targets y.
        Returns:
            A tuple of (x, y) tensors, where x is the input and y is the target.
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
