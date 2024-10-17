import torch


def zap_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Loss function that considers spikes Mean(e^(5*y_true)*(y_pred-y_true))."""
    base_loss = torch.mean(torch.exp(5 * y_true) * torch.square(y_pred - y_true))

    # Penalty for negative predictions
    negative_penalty = torch.mean(torch.relu(-y_pred))

    # Adjust the weight of the penalty as needed (e.g., 5 can be modified)
    total_loss = base_loss + 5 * negative_penalty

    return total_loss
