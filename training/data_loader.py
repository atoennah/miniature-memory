# [INJECTOR: THE PHILOSOPHY OF MEMORY-MAPPED DATA INGESTION]
#
# Training LLMs requires high-throughput data pipelines that don't exhaust system RAM.
# This DataManager implements "Memory-Mapped Tokenization" and "Vectorized Ingestion."
#
# 1. Memory Mapping (np.memmap):
#    Instead of loading a 10GB dataset into RAM, we map the file on disk directly
#    into the process's virtual address space. The OS kernel handles paging data
#    in/out of RAM as needed. This allows training on datasets larger than RAM
#    with O(1) memory overhead.
#
# 2. Vectorized Ingestion:
#    [BOLT OPTIMIZATION]: We avoid Python loops when fetching batches.
#    By using NumPy's advanced indexing `data[ix[:, None] + arange(T)]`, we perform
#    a single, massive memory copy at the C/Fortran level. This minimizes
#    Global Interpreter Lock (GIL) contention and keeps the GPU fed.
#
# 3. Persistent Metadata Caching:
#    Building the vocabulary (stoi/itos) is an O(N) operation. We cache this
#    to a `.pkl` file to ensure O(1) startup time for subsequent runs or inference.

import os
import torch
import numpy as np
import pickle
from typing import Tuple, Callable, List, Dict

class DataManager:
    """
    Manages data loading, tokenization, and batching for training.
    Optimized for memory efficiency and high-throughput ingestion.
    """

    def __init__(self, data_path: str, block_size: int, batch_size: int, device: str):
        """
        Initializes the DataManager, loading or creating the tokenized dataset.
        """
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self._initialize_tokenizer_and_data(data_path)

    def _initialize_tokenizer_and_data(self, data_path: str) -> None:
        """
        Loads cached metadata and memmap if they exist, otherwise creates them.
        """
        tokenized_data_path = data_path + ".bin"
        meta_path = data_path + "_meta.pkl"
        dtype = np.uint16 # Assumes vocab_size < 65535
        chunk_size = 10 * 1024 * 1024 # 10MB chunks

        # --- Step 1: Load or Build Metadata ---
        if os.path.exists(meta_path):
            print(f"Loading metadata from {meta_path}...")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.vocab_size = meta['vocab_size']
            stoi = meta['stoi']
            itos = meta['itos']
        else:
            print(f"Building vocabulary from {data_path}...")
            char_set = set()
            with open(data_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk: break
                    char_set.update(chunk)

            chars = sorted(list(char_set))
            self.vocab_size = len(chars)
            stoi = {ch: i for i, ch in enumerate(chars)}
            itos = {i: ch for i, ch in enumerate(chars)}

            print(f"Saving metadata to {meta_path}...")
            with open(meta_path, 'wb') as f:
                pickle.dump({'vocab_size': self.vocab_size, 'stoi': stoi, 'itos': itos}, f)

        self.encode = lambda s: [stoi.get(c, 0) for c in s]
        self.decode = lambda l: ''.join([itos.get(i, '') for i in l])

        # --- Step 2: Load or Create Memory-Mapped Data ---
        if not os.path.exists(tokenized_data_path):
            print(f"Tokenizing data to {tokenized_data_path} (chunked)...")
            # Count total characters first to allocate memmap
            total_tokens = 0
            with open(data_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk: break
                    total_tokens += len(chunk)

            mm = np.memmap(tokenized_data_path, dtype=dtype, mode='w+', shape=(total_tokens,))

            # Write in chunks to save RAM
            curr_idx = 0
            with open(data_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk: break
                    encoded = self.encode(chunk)
                    mm[curr_idx:curr_idx+len(encoded)] = encoded
                    curr_idx += len(encoded)
            mm.flush()
            del mm

        self.data = np.memmap(tokenized_data_path, dtype=dtype, mode='r')
        print(f"Dataset initialized. Total tokens: {len(self.data)}")

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a batch of data using vectorized NumPy indexing for speed.
        """
        ix = np.random.randint(0, len(self.data) - self.block_size, (self.batch_size,))
        offsets = np.arange(self.block_size)
        inds = ix[:, None] + offsets

        x_np = self.data[inds].astype(np.int64)
        y_np = self.data[inds + 1].astype(np.int64)

        x = torch.from_numpy(x_np).to(self.device)
        y = torch.from_numpy(y_np).to(self.device)

        return x, y
