import torch
import torch.nn as nn


class LSTM(nn.Module):
    """Super scuffed LSTM."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_stacked_layers: int,
    ) -> None:
        """Initialize the LSTM and its layers."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_stacked_layers, batch_first=True
        )

        # self.fc1 = nn.Linear((input_size * hidden_size), hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 24)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Define the forward pass."""
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size)
        self.lstm.flatten_parameters()

        x, _ = self.lstm(x, (h0, c0))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x[:, -1, :])

        return x
