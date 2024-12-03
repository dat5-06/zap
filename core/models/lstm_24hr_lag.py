import torch
import torch.nn as nn
from core.models.abstract_rnn import RNNBaseClass


class LSTM24hrLag(RNNBaseClass):
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

        self.fc = nn.Linear(hidden_size + 24, horizon)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Define the forward pass."""
        batch_size = x.size(0)
        # x has shape (batch_size, lookback, input_size)
        # We want to get the consumption for the last 24 hours
        # -24 is the last 24 hours and -1 is the consumption column
        lag = x[:, -24:, -1].squeeze()

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        # The out tensor has the output of each memory cell in the LSTM
        # It has shape (batch_size, lookback/LSTM_memory_cells, hidden_size)
        out = out[:, -1, :]  # We only want output of the last LSTM memory cell

        # After the LSTM layer, the output is concatenated with the 24hr lag
        out = torch.cat((out, lag), dim=1)
        out = self.fc(out)
        return out
