"""Unit tests. Run: pytest tests/"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.cnn import ECGCNN, count_parameters


def test_model_output_shape():
    model = ECGCNN(num_classes=5)
    x = torch.randn(8, 360)  # 2D input
    out = model(x)
    assert out.shape == (8, 5)


def test_model_output_shape_3d_input():
    model = ECGCNN(num_classes=5)
    x = torch.randn(8, 1, 360)  # 3D input
    out = model(x)
    assert out.shape == (8, 5)


def test_model_param_count_under_limit():
    """Keep the model small enough for edge inference."""
    model = ECGCNN()
    assert count_parameters(model) < 200_000, "Model too large for edge deployment"


def test_model_deterministic_with_seed():
    """Same seed → same weights → same eval output (dropout off)."""
    torch.manual_seed(42)
    m1 = ECGCNN()
    m1.eval()
    torch.manual_seed(42)
    m2 = ECGCNN()
    m2.eval()
    x = torch.randn(2, 360)
    with torch.no_grad():
        assert torch.allclose(m1(x), m2(x))


def test_model_handles_different_batch_sizes():
    model = ECGCNN()
    model.eval()
    for bs in [1, 4, 16, 64]:
        x = torch.randn(bs, 360)
        out = model(x)
        assert out.shape[0] == bs
