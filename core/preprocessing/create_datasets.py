import torch
from torch.utils.data import Dataset
from util.util import read_csv


# TODO: Create one dataset for each csv then concatenate or one dataset with all csv?
# TODO: Use a transform(?) and/or split and normalize data first?
class TreforDataset(Dataset):
    """Dataset for Trefor."""

    def __init__(self, file_path: str) -> None:
        """Initialize dataset.

        Arguments:
            file_path: Path of the preprocessed trefor data

        """
        self.data = read_csv(file_path).to_numpy()
        self.X = torch.from_numpy(self.data[:, 1:])
        self.y = torch.from_numpy(self.data[:, :1])

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.data)

    def __getitem__(self: "TreforDataset", index: int) -> tuple:
        """Return tuple from dataset."""
        return self.X[index, :], self.y[index, :]
