import torch
import torch.nn as nn
from core.models.abstract_rnn import RNNBaseClass


class CNNLSTM24hrLag(RNNBaseClass):
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
        self.fc = nn.Linear((hidden_size + 24), horizon)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Define the forward pass."""
        # x has shape (batch_size, lookback, input_size)
        # We want to get the consumption for the last 24 hours
        # -24 is the last 24 hours and -1 is the consumption column
        lag = x[:, -24:, -1].squeeze()

        # CNN expects input in (batch_size, input_size, lookback) format
        x_cnn = x.permute(0, 2, 1)  # (batch_size, input_size, lookback)
        x_cnn = self.cnn(x_cnn)  # (batch_size, 64, reduced_lookback)

        # Initialize cell state and hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        # Flatten for LSTM
        x_cnn = x_cnn.permute(0, 2, 1)  # (batch_size, reduced_lookback, 64)
        out, _ = self.lstm(
            x_cnn, (h0, c0)
        )  # (batch_size, reduced_lookback, hidden_size)

        # The out tensor has the output of each memory cell in the LSTM
        # It has shape (batch_size, lookback/LSTM_memory_cells, hidden_size)
        out = out[:, -1, :]  # We only want output of the last LSTM memory cell

        # After the LSTM layer, the output is concatenated with the 24hr lag
        out = torch.cat((out, lag), dim=1)

        # Fully connected layers
        x = self.fc(out)  # (batch_size, 100)

        return x
