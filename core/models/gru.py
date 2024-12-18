import torch
import torch.nn as nn
from core.models.abstract_rnn import RNNBaseClass


class GRU(RNNBaseClass):
    """Simple GRU implementation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float,
        horizon: int,
        lookback: int,
    ) -> None:
        """Initialize the GRU and its layers."""
        super().__init__(
            input_size, hidden_size, num_layers, dropout_rate, horizon, lookback
        )

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )
        self.fully_connected = nn.Linear(hidden_size, horizon)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass for the model."""
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fully_connected(out)
        return out
