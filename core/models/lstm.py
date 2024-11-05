import torch
import torch.nn as nn
import warnings

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class LSTM(nn.Module):
    """LSTM model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float,
    ) -> None:
        """Initialize the LSTM and its layers."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Suppress user warning from dropout
        warnings.filterwarnings("ignore")

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )

        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, 24)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Define the forward pass."""
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        x, _ = self.lstm(x, (h0, c0))
        x = self.relu(x)
        x = self.fc2(x[:, -1, :])

        return x

    def get_members(self) -> list:
        """Get all members used to initialise the object."""
        return [self.input_size, self.hidden_size, self.num_layers, self.dropout_rate]
