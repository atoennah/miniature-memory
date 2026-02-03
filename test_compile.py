
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 10)
    def forward(self, x):
        return self.lin(x)

model = SimpleModel()
try:
    compiled_model = torch.compile(model)
    x = torch.randn(1, 10)
    y = compiled_model(x)
    print("torch.compile works!")
except Exception as e:
    print(f"torch.compile failed: {e}")
