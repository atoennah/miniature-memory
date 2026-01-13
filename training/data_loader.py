"""
Manages data loading, tokenization, and batching for model training.

This module is optimized for memory efficiency by memory-mapping the tokenized
dataset, allowing for the training of models on datasets that are much
larger than the available RAM.
"""
import os
import torch
import numpy as np
from typing import Tuple, Callable, List, Dict

class DataManager:
    """
    Handles data loading, tokenization, and batching from a single text file.

    This class reads a text file, creates a character-level vocabulary, and
    splits the data into training and validation sets. Both splits are stored
    as memory-mapped numpy arrays for efficient data loading.

    Attributes:
        train_data (np.memmap): Memory-mapped array for the training split.
        val_data (np.memmap): Memory-mapped array for the validation split.
        vocab_size (int): The number of unique characters in the vocabulary.
        encode (Callable[[str], List[int]]): A function to encode a string
            into a list of token IDs.
        decode (Callable[[List[int]], str]): A function to decode a list of
            token IDs into a string.
        block_size (int): The context length for the model.
        batch_size (int): The number of sequences in a batch.
        device (str): The device to move tensors to ('cpu' or 'cuda').
    """

    def __init__(self, data_path: str, block_size: int, batch_size: int, device: str):
        """
        Initializes the DataManager, tokenizes the data, and memory-maps it.

        Args:
            data_path (str): Path to the training data file (e.g., 'train.txt').
            block_size (int): The context length for the transformer model.
            batch_size (int): The number of independent sequences in a batch.
            device (str): The computing device ('cuda' or 'cpu').
        """
        self.block_size: int = block_size
        self.batch_size: int = batch_size
        self.device: str = device
        self._initialize_tokenizer_and_data(data_path)

    def _initialize_tokenizer_and_data(self, data_path: str) -> None:
        """
        Reads data, creates a tokenizer, and prepares memory-mapped train/val splits.

        This private method performs the following steps:
        1. Reads the raw text data.
        2. Creates a character-level vocabulary and tokenizer functions.
        3. Splits the data into a 90% training set and a 10% validation set.
        4. Tokenizes each split and saves them to separate `.bin` files using
           memory-mapping for efficiency.

        Args:
            data_path (str): The path to the input text file.
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            text: str = f.read()

        chars: List[str] = sorted(list(set(text)))
        self.vocab_size: int = len(chars)

        stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        self.encode: Callable[[str], List[int]] = lambda s: [stoi.get(c, 0) for c in s]
        self.decode: Callable[[List[int]], str] = lambda l: ''.join([itos.get(i, '') for i in l])

        # Tokenize and memory-map the data in chunks
        self._process_and_tokenize_chunked(data_path, stoi)

    def _process_and_tokenize_chunked(self, data_path: str, stoi: Dict[str, int]) -> None:
        """
        Tokenizes a text file in chunks and creates memory-mapped train/val splits.

        This method is designed to handle files larger than RAM by reading,
        tokenizing, and writing the data in manageable chunks.

        Args:
            data_path (str): The path to the input text file.
            stoi (Dict[str, int]): The vocabulary mapping characters to integers.
        """
        train_token_path: str = data_path + ".train.bin"
        val_token_path: str = data_path + ".val.bin"
        dtype = np.uint16

        # Estimate file sizes
        n = os.path.getsize(data_path)
        train_size = int(n * 0.9)
        val_size = n - train_size

        # Create memory-mapped files
        train_mm = np.memmap(train_token_path, dtype=dtype, mode='w+', shape=(train_size,))
        val_mm = np.memmap(val_token_path, dtype=dtype, mode='w+', shape=(val_size,))

        # Process the file in chunks
        chunk_size = 100 * 1024 * 1024  # 100MB
        train_ptr, val_ptr = 0, 0
        with open(data_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                # Encode the chunk
                encoded_chunk = [stoi.get(c, 0) for c in chunk]

                # Determine split point
                chunk_train_size = int(len(encoded_chunk) * 0.9)

                # Write to train and val memory-mapped files
                train_chunk = encoded_chunk[:chunk_train_size]
                val_chunk = encoded_chunk[chunk_train_size:]

                train_mm[train_ptr:train_ptr + len(train_chunk)] = train_chunk
                val_mm[val_ptr:val_ptr + len(val_chunk)] = val_chunk

                train_ptr += len(train_chunk)
                val_ptr += len(val_chunk)

        # Trim the memory-mapped files to their actual size
        train_mm = np.memmap(train_token_path, dtype=dtype, mode='r+', shape=(train_ptr,))
        val_mm = np.memmap(val_token_path, dtype=dtype, mode='r+', shape=(val_ptr,))

        train_mm.flush()
        val_mm.flush()

        self.train_data: np.memmap = np.memmap(train_token_path, dtype=dtype, mode='r')
        self.val_data: np.memmap = np.memmap(val_token_path, dtype=dtype, mode='r')


    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a batch of data from either the training or validation split.

        This method randomly selects starting indices from the specified data split
        and creates input (`x`) and target (`y`) tensors of shape
        (batch_size, block_size).

        Args:
            split (str): The data split to use, either 'train' or 'val'.

        Returns:
            A tuple containing input (x) and target (y) tensors, moved to the
            configured device.
        """
        data: np.memmap = self.train_data if split == 'train' else self.val_data
        ix: torch.Tensor = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x: torch.Tensor = torch.stack([torch.from_numpy(data[i:i+self.block_size].astype(np.int64)) for i in ix])
        y: torch.Tensor = torch.stack([torch.from_numpy(data[i+1:i+self.block_size+1].astype(np.int64)) for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
