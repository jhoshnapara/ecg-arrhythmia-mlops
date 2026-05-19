"""PyTorch Dataset wrappers for the preprocessed MIT-BIH data."""

import numpy as np
import torch
from torch.utils.data import Dataset


class ECGBeatDataset(Dataset):
    """Loads preprocessed beats from a .npz file."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data["X"]).float()
        self.y = torch.from_numpy(data["y"]).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.y.numpy(), minlength=5)
