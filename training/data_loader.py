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
        """
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
