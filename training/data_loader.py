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
        """Reads data, creates a tokenizer, and prepares the memory-mapped dataset."""
        # Read the entire text to build the vocabulary
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}

        self.encode: Callable[[str], List[int]] = lambda s: [stoi.get(c, 0) for c in s]
        self.decode: Callable[[List[int]], str] = lambda l: ''.join([itos.get(i, '') for i in l])

        # Define the path for the memory-mapped file
        tokenized_data_path = data_path + ".bin"

        # Create a memory-mapped file to store the tokenized data
        # Using uint16 for tokens, assuming vocab_size < 65535
        dtype = np.uint16
        mm = np.memmap(tokenized_data_path, dtype=dtype, mode='w+', shape=(len(text),))

        # Encode the text and write it to the memory-mapped file
        # This is done in chunks to avoid loading the full encoded list into memory
        chunk_size = 100 * 1024 * 1024  # 100MB chunks
        i = 0
        while i < len(text):
            chunk = text[i:i+chunk_size]
            encoded_chunk = self.encode(chunk)
            mm[i:i+len(encoded_chunk)] = encoded_chunk
            i += len(chunk)


        # Flush changes to disk and set the data attribute
        mm.flush()
        self.data = np.memmap(tokenized_data_path, dtype=dtype, mode='r')

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

        # Create input and target sequences by reading from the memmap array
        # and converting to torch tensors
        x = torch.stack([torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i+1:i+self.block_size+1].astype(np.int64)) for i in ix])

        # Move tensors to the specified device
        x, y = x.to(self.device), y.to(self.device)
        return x, y
