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
    """

    def __init__(self, data_path: str, block_size: int, batch_size: int, device: str):
        """
        Initializes the DataManager, tokenizes the data (if needed), and memory-maps it.
        """
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self._initialize_tokenizer_and_data(data_path)

    def _initialize_tokenizer_and_data(self, data_path: str) -> None:
        """
        Loads or creates tokenized data and metadata.
        """
        bin_path = data_path + ".bin"
        meta_path = data_path + "_meta.pkl"

        if os.path.exists(bin_path) and os.path.exists(meta_path):
            print(f"Loading existing tokenized data from {bin_path}")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.vocab_size = meta['vocab_size']
            stoi = meta['stoi']
            itos = meta['itos']
            self.encode = lambda s: [stoi.get(c, 0) for c in s]
            self.decode = lambda l: ''.join([itos.get(i, '') for i in l])

            # Get size from file
            file_size = os.path.getsize(bin_path)
            # Assuming uint16 (2 bytes)
            total_size = file_size // 2
            self.data = np.memmap(bin_path, dtype=np.uint16, mode='r', shape=(total_size,))
        else:
            print(f"Tokenizing raw data from {data_path}...")
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

            self.encode: Callable[[str], List[int]] = lambda s: [stoi.get(c, 0) for c in s]
            self.decode: Callable[[List[int]], str] = lambda l: ''.join([itos.get(i, '') for i in l])

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
            print(f"Tokenization complete. Saved to {bin_path}")

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a small batch of data. Optimized with NumPy vectorization.
        """
        ix = np.random.randint(0, len(self.data) - self.block_size, (self.batch_size,))

        # Use a single numpy allocation and loop for memory efficiency and speed
        x_np = np.stack([self.data[i:i+self.block_size] for i in ix]).astype(np.int64)
        y_np = np.stack([self.data[i+1:i+self.block_size+1] for i in ix]).astype(np.int64)

        x = torch.from_numpy(x_np).to(self.device)
        y = torch.from_numpy(y_np).to(self.device)

        return x, y
