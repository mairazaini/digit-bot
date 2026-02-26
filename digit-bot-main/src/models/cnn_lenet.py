# src/models/cnn_lenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNetMNIST(nn.Module):
    """
    Simple CNN for MNIST:
    (1x28x28) -> Conv -> Pool -> Conv -> Pool -> FC -> FC -> (10 classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # After two pools: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (N,32,14,14)
        x = self.pool(F.relu(self.conv2(x)))  # (N,64,7,7)
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
