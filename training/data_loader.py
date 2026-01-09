import torch

class DataManager:
    """Manages data loading, tokenization, and batching."""
    def __init__(self, data_path, block_size, batch_size, device):
        self.data_path = data_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        # Read the text file once.
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()

        self.vocab_size, self.encode, self.decode = self._build_tokenizer()
        self.data = self._load_data()

    def _load_data(self):
        """Encodes the entire text dataset."""
        return torch.tensor(self.encode(self.text), dtype=torch.long)

    def _build_tokenizer(self):
        """Creates a simple char-level tokenizer from the text."""
        chars = sorted(list(set(self.text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        return vocab_size, encode, decode

    def get_batch(self):
        """Generates a small batch of data of inputs x and targets y."""
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
