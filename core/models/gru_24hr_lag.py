import torch
import torch.nn as nn
from core.models.abstract_rnn import RNNBaseClass


class GRU24hrLag(RNNBaseClass):
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
        self.fc = nn.Linear(hidden_size + 24, horizon)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass for the model."""
        # x has shape (batch_size, lookback, input_size)
        # We want to get the consumption for the last 24 hours
        # -24 is the last 24 hours and -1 is the consumption column
        lag = x[:, -24:, -1].squeeze()

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.gru(x, h0)
        # The out tensor has the output of each cell in the GRU
        # It has shape (batch_size, lookback/GRU_memory_cells, hidden_size)
        out = out[:, -1, :]  # We only want output of the last LSTM memory cell

        # After the GRU layer, the output is concatenated with the 24hr lag
        out = torch.cat((out, lag), dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
