"""
Multi-Layer Perceptron (MLP) implementation using PyTorch
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    3-layer MLP as specified in assignment:
    - Linear(50, 512) - ReLU
    - Linear(512, 512) - BatchNorm(512) - ReLU
    - Linear(512, 10)
    """

    def __init__(self):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(50, 512)
        self.relu1 = nn.ReLU()

        self.layer2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()

        self.layer3 = nn.Linear(512, 10)

    def forward(self, x):
        # Layer 1
        x = self.layer1(x)
        x = self.relu1(x)

        # Layer 2
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Layer 3 (output)
        x = self.layer3(x)

        return x


class MLPDeep(nn.Module):
    """
    Deeper MLP variant with 5 layers for depth experiments.
    """

    def __init__(self):
        super(MLPDeep, self).__init__()

        self.layer1 = nn.Linear(50, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()

        self.layer2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()

        self.layer3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()

        self.layer4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()

        self.layer5 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.layer1(x)))
        x = self.relu2(self.bn2(self.layer2(x)))
        x = self.relu3(self.bn3(self.layer3(x)))
        x = self.relu4(self.bn4(self.layer4(x)))
        x = self.layer5(x)
        return x


class MLPWide(nn.Module):
    """
    Wider MLP variant with larger hidden layers for size experiments.
    """

    def __init__(self):
        super(MLPWide, self).__init__()

        self.layer1 = nn.Linear(50, 1024)
        self.relu1 = nn.ReLU()

        self.layer2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu2 = nn.ReLU()

        self.layer3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.bn2(self.layer2(x)))
        x = self.layer3(x)
        return x


class MLPNarrow(nn.Module):
    """
    Narrower MLP variant with smaller hidden layers for size experiments.
    """

    def __init__(self):
        super(MLPNarrow, self).__init__()

        self.layer1 = nn.Linear(50, 128)
        self.relu1 = nn.ReLU()

        self.layer2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.bn2(self.layer2(x)))
        x = self.layer3(x)
        return x
