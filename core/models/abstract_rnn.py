import torch
import torch.nn as nn
import warnings
from abc import ABC, abstractmethod


class RNNBaseClass(nn.Module, ABC):
    """Abtract class for RNN models."""

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
        super(RNNBaseClass, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.horizon = horizon
        self.lookback = lookback
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Suppress user warning from dropout
        warnings.filterwarnings("ignore")

    @abstractmethod
    def forward(self, x: torch.tensor) -> torch.tensor:
        """Abtract forward pass for the model."""
        pass

    def get_members(self) -> list:
        """Abstract method impl. to get all members used to initialise the object."""
        return [
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.dropout_rate,
            self.horizon,
            self.lookback,
        ]
