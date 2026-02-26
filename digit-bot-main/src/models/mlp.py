# src/models/mlp.py
import torch.nn as nn

class MLP(nn.Module):
    """
    Simple MLP for MNIST:
    - Flatten 28x28 -> 784
    - Two hidden layers
    - Output 10 classes
    """
    def __init__(self, input_dim: int = 28 * 28, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                 # (N,1,28,28) -> (N,784)
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)
