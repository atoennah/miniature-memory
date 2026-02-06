import torch
import numpy as np
import pickle
import os
from typing import Tuple, Callable, List, Dict, Optional

class DataManager:
    """
    Manages data loading, tokenization, and batching for training.

    This class is optimized for memory efficiency and performance. It memory-maps
    the tokenized dataset and caches results in .bin and _meta.pkl files to avoid
    redundant processing. It uses vectorized batch extraction for improved throughput.

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
        Initializes the DataManager, tokenizes the data (or loads from cache),
        and sets up the memory-mapped dataset.

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
        Handles tokenization with a caching mechanism. If cached artifacts exist,
        it loads them; otherwise, it processes the raw text file in chunks.
        """
        bin_path = data_path + ".bin"
        meta_path = data_path + "_meta.pkl"

        if os.path.exists(bin_path) and os.path.exists(meta_path):
            print(f"DataManager: Loading cached tokenized data from {bin_path}")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)

            self.vocab_size = meta['vocab_size']
            stoi = meta['stoi']
            itos = meta['itos']

            # Load the existing memory-mapped file
            self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        else:
            print(f"DataManager: No cache found. Tokenizing {data_path}...")

            # --- Step 1: Build vocabulary by streaming the file ---
            char_set = set()
            total_size = 0
            chunk_size = 10 * 1024 * 1024  # 10MB chunks for vocab building

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

            # --- Step 2: Create memory-mapped file and tokenize in chunks ---
            dtype = np.uint16  # Assuming vocab_size < 65535
            mm = np.memmap(bin_path, dtype=dtype, mode='w+', shape=(total_size,))

            # Process the file again, this time tokenizing and writing to the memmap
            processed_size = 0
            with open(data_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    # Inline encoding for speed in this specific loop
                    encoded_chunk = [stoi.get(c, 0) for c in chunk]
                    mm[processed_size:processed_size + len(encoded_chunk)] = encoded_chunk
                    processed_size += len(encoded_chunk)

            mm.flush()
            self.data = mm # Use the just-created memmap

            # --- Step 3: Save metadata for future runs ---
            meta = {
                'vocab_size': self.vocab_size,
                'itos': itos,
                'stoi': stoi,
            }
            with open(meta_path, 'wb') as f:
                pickle.dump(meta, f)
            print(f"DataManager: Tokenization complete. Cache saved to {bin_path} and {meta_path}")

        # Set up the public encode/decode functions
        stoi_cached = stoi
        itos_cached = itos
        self.encode: Callable[[str], List[int]] = lambda s: [stoi_cached.get(c, 0) for c in s]
        self.decode: Callable[[List[int]], str] = lambda l: ''.join([itos_cached.get(i, '') for i in l])

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a batch of inputs (x) and targets (y) using vectorized
        NumPy extraction for maximum throughput.
        """
        # Generate random starting indices for the batch
        ix = np.random.randint(0, len(self.data) - self.block_size, (self.batch_size,))

        # Vectorized extraction via NumPy stack
        # This avoids multiple calls to torch.from_numpy within a loop
        x_np = np.stack([self.data[i:i+self.block_size] for i in ix])
        y_np = np.stack([self.data[i+1:i+self.block_size+1] for i in ix])

        # Convert to torch tensors and move to device in one go
        x = torch.from_numpy(x_np.astype(np.int64)).to(self.device)
        y = torch.from_numpy(y_np.astype(np.int64)).to(self.device)

        return x, y
