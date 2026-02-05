import torch
import numpy as np
import os
import pickle
from typing import Tuple, Callable, List, Dict

class DataManager:
    """
    Manages data loading, tokenization, and batching for training.

    This class is optimized for memory efficiency. It memory-maps the tokenized
    dataset, allowing for the training of models on datasets that are much
    larger than the available RAM.

    Attributes:
        data (np.memmap): Memory-mapped array of token IDs.
        vocab_size (int): The number of unique characters in the vocabulary.
        encode (callable): A function to encode a string into a list of token IDs.
        decode (callable): A function to decode a list of token IDs into a string.
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
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self._initialize_tokenizer_and_data(data_path)

    def _initialize_tokenizer_and_data(self, data_path: str) -> None:
        """
        Reads data in chunks, creates a tokenizer, and prepares a memory-mapped
        dataset without loading the entire file into RAM. Caches results to disk.
        """
        bin_path = data_path + ".bin"
        meta_path = data_path + "_meta.pkl"

        if os.path.exists(bin_path) and os.path.exists(meta_path):
            print(f"Loading cached dataset from {bin_path}...")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.vocab_size = meta['vocab_size']
            stoi, itos = meta['stoi'], meta['itos']
            self.encode = lambda s: [stoi.get(c, 0) for c in s]
            self.decode = lambda l: ''.join([itos.get(i, '') for i in l])
            self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
            return

        print(f"Tokenizing dataset {data_path}...")
        # --- Step 1: Build vocabulary by streaming the file ---
        char_set = set()
        total_size = 0
        chunk_size = 10 * 1024 * 1024

        with open(data_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                char_set.update(chunk)
                total_size += len(chunk)

        chars = sorted(list(char_set))
        self.vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        self.encode = lambda s: [stoi.get(c, 0) for c in s]
        self.decode = lambda l: ''.join([itos.get(i, '') for i in l])

        # Save metadata
        with open(meta_path, 'wb') as f:
            pickle.dump({'vocab_size': self.vocab_size, 'stoi': stoi, 'itos': itos}, f)

        # --- Step 2: Create memory-mapped file and tokenize in chunks ---
        dtype = np.uint16
        mm = np.memmap(bin_path, dtype=dtype, mode='w+', shape=(total_size,))

        processed_size = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                encoded_chunk = self.encode(chunk)
                mm[processed_size:processed_size + len(encoded_chunk)] = encoded_chunk
                processed_size += len(encoded_chunk)

        mm.flush()
        self.data = np.memmap(bin_path, dtype=dtype, mode='r', shape=(total_size,))

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a small batch of data of inputs x and targets y.
        Uses vectorized NumPy indexing for maximum throughput.
        """
        # Generate random starting indices
        ix = np.random.randint(0, len(self.data) - self.block_size, (self.batch_size,))

        # Vectorized extraction: create a 2D array of indices
        # Shape: (batch_size, block_size)
        offsets = np.arange(self.block_size)
        ix_x = ix[:, None] + offsets[None, :]
        ix_y = ix_x + 1

        # Index into memmap with 2D array of indices
        # Use .astype(np.int64) before converting to torch to ensure compatibility
        x_np = self.data[ix_x].astype(np.int64)
        y_np = self.data[ix_y].astype(np.int64)

        x = torch.from_numpy(x_np).to(self.device)
        y = torch.from_numpy(y_np).to(self.device)

        return x, y
