# [INJECTOR: THE LOGOS OF MEMORY-MAPPED DATASETS]
#
# Training LLMs on massive datasets presents a fundamental challenge: how do we
# access billions of tokens without exhausting the system's RAM?
#
# This module implements a "Memory-Mapped Tokenization" strategy using NumPy's
# `memmap`. This technique maps a file on disk directly into the model's
# virtual address space.
#
# WHY MEMMAP?
# 1. O(1) RANDOM ACCESS: We can jump to any offset in a multi-gigabyte file
#    instantly without reading the preceding bytes.
# 2. O(N) MEMORY EFFICIENCY: The OS manages a page cache, loading only the
#    required segments of the file into physical RAM. This allows us to
#    train on datasets far larger than the available memory.
# 3. ZERO-COPY DESERIALIZATION: The data is already in its binary,
#    tensor-ready format on disk. No expensive JSON parsing or string
#    manipulation is required during the training loop.
#
# The pipeline follows a strict two-pass architecture:
# - PASS 1 (VOCAB): Streams the raw text to build a unique character-level
#   vocabulary.
# - PASS 2 (TOKENIZE): Streams the text again, encoding characters into
#   integers and flushing them directly to the `.bin` memory map in chunks.

import torch
import numpy as np
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
        # [INJECTOR NOTE]: We use a streaming approach with 10MB chunks to
        # build the vocabulary. This ensures that even for 100GB+ files,
        # our RAM usage remains constant and minimal.
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

        # --- Step 2: Create memory-mapped file and tokenize in chunks ---
        # [INJECTOR NOTE]: We use uint16 for the token storage. This is a
        # compromise that supports vocab sizes up to 65,535 while using
        # only 2 bytes per token, halving the disk footprint compared to
        # standard 32-bit integers.
        tokenized_data_path = data_path + ".bin"
        dtype = np.uint16  # Assuming vocab_size < 65535

        # Create the memory-mapped file with the correct total size
        mm = np.memmap(tokenized_data_path, dtype=dtype, mode='w+', shape=(total_size,))

        # Process the file again, this time tokenizing and writing to the memmap
        # [INJECTOR NOTE]: Chunked writing to the memmap is critical. It
        # allows the OS to flush pages to disk asynchronously, preventing
        # I/O bottlenecks.
        processed_size = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                encoded_chunk = self.encode(chunk)
                mm[processed_size:processed_size + len(encoded_chunk)] = encoded_chunk
                processed_size += len(encoded_chunk)

        # Flush changes to disk and set the data attribute for reading
        mm.flush()
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
        # [INJECTOR NOTE]: Random sampling from the memmap is essentially
        # "free" in terms of CPU overhead, as the OS handles the page faults
        # and pre-fetching in the background.

        # Generate random starting indices for each batch
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))

        # Create input and target sequences by reading from the memmap array
        # and converting to torch tensors
        x = torch.stack([torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i+1:i+self.block_size+1].astype(np.int64)) for i in ix])

        # Move tensors to the specified device
        x, y = x.to(self.device), y.to(self.device)
        return x, y
