import torch


def zap_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Loss function that considers spikes Mean(e^(5*y_true)*(y_pred-y_true))."""
    return torch.mean(torch.exp(5 * y_true) * torch.square(y_pred - y_true))
