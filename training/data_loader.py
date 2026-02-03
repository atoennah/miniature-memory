import torch
import numpy as np
import os
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
        dataset without loading the entire file into RAM.
        """
        # --- Step 1: Build vocabulary by streaming the file ---
        char_set = set()
        total_size = 0
        chunk_size = 10 * 1024 * 1024  # 10MB chunks for vocab building

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                char_set.update(chunk)
                total_size += len(chunk)

        chars = sorted(list(char_set))
        self.vocab_size = len(chars)

        stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}

        self.encode: Callable[[str], List[int]] = lambda s: [stoi.get(c, 0) for c in s]
        self.decode: Callable[[List[int]], str] = lambda l: ''.join([itos.get(i, '') for i in l])

        # --- Step 2: Create memory-mapped file and tokenize in chunks ---
        tokenized_data_path = data_path + ".bin"
        dtype = np.uint16  # Assuming vocab_size < 65535

        # Bolt Optimization: Skip re-tokenization if the bin file is already present and matches expected size
        expected_size = total_size * np.dtype(dtype).itemsize
        if os.path.exists(tokenized_data_path) and os.path.getsize(tokenized_data_path) == expected_size:
            print(f"⚡ Bolt: Utilizing existing tokenized artifact: {tokenized_data_path}")
        else:
            print(f"⚡ Bolt: Tokenizing dataset into {tokenized_data_path}...")
            # Create the memory-mapped file with the correct total size
            mm = np.memmap(tokenized_data_path, dtype=dtype, mode='w+', shape=(total_size,))

            # Process the file again, this time tokenizing and writing to the memmap
            processed_size = 0
            with open(data_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    encoded_chunk = self.encode(chunk)
                    mm[processed_size:processed_size + len(encoded_chunk)] = encoded_chunk
                    processed_size += len(encoded_chunk)

            # Flush changes to disk
            mm.flush()

        # Set the data attribute for reading
        self.data = np.memmap(tokenized_data_path, dtype=dtype, mode='r', shape=(total_size,))

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a small batch of data of inputs x and targets y.

        This method reads directly from the memory-mapped file.

        Returns:
            A tuple containing:
            - A (batch_size, block_size) tensor of input token IDs.
            - A (batch_size, block_size) tensor of target token IDs.
        """
        # Generate random starting indices for each batch
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))

        # Bolt Optimization: Vectorized extraction from memmap via NumPy stacking
        # This is more efficient than looping over torch.from_numpy calls
        x_np = np.stack([self.data[i:i+self.block_size] for i in ix])
        y_np = np.stack([self.data[i+1:i+self.block_size+1] for i in ix])

        x = torch.from_numpy(x_np.astype(np.int64)).to(self.device)
        y = torch.from_numpy(y_np.astype(np.int64)).to(self.device)

        return x, y
