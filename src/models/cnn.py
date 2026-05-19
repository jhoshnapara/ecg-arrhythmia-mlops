"""
1D CNN for single-beat arrhythmia classification.

Architecture (~80K params, small enough for CPU inference on edge devices):
- 3 conv blocks (Conv1d + BN + ReLU + MaxPool)
- Global average pooling
- 2 dense layers with dropout
- 5-class softmax output

Design notes:
- Input: (batch, 1, 360) — 1-second window @ 360 Hz
- BatchNorm before activation = standard ResNet-style block
- GAP instead of flatten — fewer params, less overfitting
- Dropout in dense layers only — conv layers already regularized by BN
"""

import torch
import torch.nn as nn


class ECGCNN(nn.Module):
    """Compact 1D CNN for ECG beat classification."""

    def __init__(self, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1 -> 32 channels, downsample 360 -> 180
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # Block 2: 32 -> 64, downsample 180 -> 90
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # Block 3: 64 -> 128, downsample 90 -> 45
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # Global avg pool: (B, 128, 45) -> (B, 128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 360) or (B, 1, 360)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.features(x)
        return self.classifier(features)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    model = ECGCNN()
    x = torch.randn(4, 1, 360)
    out = model(x)
    assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"
    print(f"Model output shape: {out.shape}")
    print(f"Trainable params: {count_parameters(model):,}")
