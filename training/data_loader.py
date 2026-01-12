import os
import torch
import numpy as np
from typing import Tuple, Callable, List, Dict

class DataManager:
    """
    Manages data loading, tokenization, and batching for training and validation.

    This class is optimized for memory efficiency. It splits the dataset into
    training and validation sets, then memory-maps the tokenized data, allowing
    for the training of models on datasets much larger than available RAM.

    Attributes:
        train_data (np.memmap): Memory-mapped array for the training dataset.
        val_data (np.memmap): Memory-mapped array for the validation dataset.
        vocab_size (int): The number of unique characters in the vocabulary.
        encode (callable): Encodes a string into a list of token IDs.
        decode (callable): Decodes a list of token IDs into a string.
        block_size (int): The context length for the model.
        batch_size (int): The number of sequences in a batch.
    """

    def __init__(self, data_path: str, block_size: int, batch_size: int):
        """
        Initializes the DataManager, tokenizes the data, splits it,
        and creates memory-mapped files for both training and validation sets.
        """
        self.block_size = block_size
        self.batch_size = batch_size
        self._initialize_tokenizer_and_data(data_path)

    def _initialize_tokenizer_and_data(self, data_path: str) -> None:
        """
        Reads data, creates a tokenizer, splits data into train/val,
        and prepares memory-mapped datasets.
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}

        self.encode: Callable[[str], List[int]] = lambda s: [stoi.get(c, 0) for c in s]
        self.decode: Callable[[List[int]], str] = lambda l: ''.join([itos.get(i, '') for i in l])

        # Split data into training (90%) and validation (10%)
        n = len(text)
        train_text = text[:int(n * 0.9)]
        val_text = text[int(n * 0.9):]

        # Define paths for the tokenized .bin files
        base_dir, filename = os.path.split(data_path)
        train_token_path = os.path.join(base_dir, f"train_{filename}.bin")
        val_token_path = os.path.join(base_dir, f"val_{filename}.bin")

        self.train_data = self._tokenize_and_save(train_text, train_token_path)
        self.val_data = self._tokenize_and_save(val_text, val_token_path)

    def _tokenize_and_save(self, text: str, tokenized_path: str) -> np.memmap:
        """
        Encodes text and saves it to a memory-mapped binary file.
        This is done in chunks to manage memory usage.
        """
        dtype = np.uint16  # uint16 for tokens, assuming vocab_size < 65,535
        mm = np.memmap(tokenized_path, dtype=dtype, mode='w+', shape=(len(text),))

        chunk_size = 100 * 1024 * 1024  # Process 100MB chunks
        i = 0
        while i < len(text):
            chunk = text[i:i + chunk_size]
            encoded_chunk = self.encode(chunk)
            mm[i:i + len(encoded_chunk)] = encoded_chunk
            i += len(chunk)

        mm.flush()
        return np.memmap(tokenized_path, dtype=dtype, mode='r')

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a batch of data for either training or validation.

        Args:
            split (str): The dataset split to use, either 'train' or 'val'.

        Returns:
            A tuple of (inputs, targets) tensors, on the CPU.
        """
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))

        x = torch.stack([torch.from_numpy(data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + self.block_size + 1].astype(np.int64)) for i in ix])

        return x, y
