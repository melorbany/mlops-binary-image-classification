"""
Simple CNN architecture for binary image classification (Cats vs Dogs).
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Lightweight CNN for 224×224 RGB binary classification.

    Architecture:
      Conv(32) → BN → ReLU → Pool
      Conv(64) → BN → ReLU → Pool
      Conv(128) → BN → ReLU → Pool
      Conv(256) → BN → ReLU → GlobalAvgPool
      FC(512) → Dropout → FC(1)
    """

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112×112
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56×56
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28×28
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # 4×4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x  # raw logits (use BCEWithLogitsLoss)


def get_model(arch: str = "SimpleCNN", dropout: float = 0.5) -> nn.Module:
    """Factory function: return a model by architecture name."""
    if arch == "SimpleCNN":
        return SimpleCNN(dropout=dropout)
    raise ValueError(f"Unknown architecture: {arch}")
