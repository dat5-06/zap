import torch
import torch.nn as nn
from core.models.abstract_rnn import RNNBaseClass


class CNNLSTM(RNNBaseClass):
    """CNN_LSTM model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float,
        horizon: int,
        lookback: int,
    ) -> None:
        """Initialize the CNN-LSTM model."""
        super().__init__(
            input_size, hidden_size, num_layers, dropout_rate, horizon, lookback
        )

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=64,
                kernel_size=2,
                stride=1,
                padding="valid",
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=2,
                stride=1,
                padding="valid",
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )
        self.fc_lstm = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Define the forward pass."""
        # CNN expects input in (batch_size, input_size, lookback) format
        x_cnn = x.permute(0, 2, 1)  # (batch_size, input_size, lookback)
        x_cnn = self.cnn(x_cnn)  # (batch_size, 64, reduced_lookback)

        # Initialize cell state and hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        # Flatten for LSTM
        x_cnn = x_cnn.permute(0, 2, 1)  # (batch_size, reduced_lookback, 64)
        x_lstm, _ = self.lstm(
            x_cnn, (h0, c0)
        )  # (batch_size, reduced_lookback, hidden_size)

        # Fully connected layers
        x = self.fc_lstm(x_lstm[:, -1, :])  # (batch_size, 100)

        return x
