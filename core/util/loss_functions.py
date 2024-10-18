import torch


def zap_loss(y_pred: torch.tensor, y_true: torch.tensor) -> torch.tensor:
    """Loss function that considers spikes Mean(e^(5*y_true)*(y_pred-y_true))."""
    return torch.mean(
        torch.multiply(
            torch.exp(torch.multiply(5, y_true)),
            torch.square(torch.subtract(y_pred, y_true)),
        )
    )
