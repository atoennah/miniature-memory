import json

class CharacterTokenizer:
    def __init__(self, vocab=None):
        if vocab:
            self.chars = vocab
        else:
            self.chars = []
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def train(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return "".join([self.itos[i] for i in l])

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.chars, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            chars = json.load(f)
        return cls(vocab=chars)
