import torch
import torch.nn as nn
from core.models.abstract_rnn import RNNBaseClass


class LSTM(RNNBaseClass):
    """LSTM model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float,
        horizon: int,
        lookback: int,
    ) -> None:
        """Initialize the LSTM and its layers."""
        super().__init__(
            input_size, hidden_size, num_layers, dropout_rate, horizon, lookback
        )

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )

        self.leakyrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Define the forward pass."""
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc2(out[:, -1, :])
        return out
