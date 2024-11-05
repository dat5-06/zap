import torch
import torch.nn as nn

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class GRU(nn.Module):
    """Simple GRU implementation."""

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout_rate: float
    ) -> None:
        """Initialize the GRU and its layers."""
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )
        self.nl = nn.LeakyReLU()
        self.fc = nn.Linear((hidden_size * 24), 24)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass for the model."""
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.nl(out)
        out = self.fc(out)
        return out

    def get_members(self) -> list:
        """Get all members used to initialise the object."""
        return [self.input_size, self.hidden_size, self.num_layers]
