from torch.utils.data import Dataset
import torch


class TreforData(Dataset):
    """Initialize Trefor dataset."""

    def __init__(self, x: torch.tensor, y: torch.tensor, device: str) -> None:
        """Initialize dataset.

        Arguments:
            x: feature as torch
            y: target as torch
            device: device being used

        """
        self.x = x.to(device)
        self.y = y.to(device)

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.x)

    def __getitem__(self, i: int) -> tuple:
        """Return tuple from dataset."""
        return self.x[i], self.y[i]
