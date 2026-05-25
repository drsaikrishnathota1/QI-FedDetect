"""
models.py — Local MLP classifier for QI-FedDetect.

Architecture: 3-layer MLP with ReLU activations.
Input dimension matches QIFE output (2^q Pauli observables).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """3-layer MLP used by every client as the local intrusion detection model."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_model(input_dim: int, num_classes: int, hidden_dim: int = 128) -> MLPClassifier:
    return MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
