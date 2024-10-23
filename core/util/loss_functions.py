import torch


def zap_loss(y_pred: torch.tensor, y_true: torch.tensor) -> torch.tensor:
    """Loss function that considers spikes Mean(e^(5*y_true)*(y_pred-y_true))."""
    base_loss = torch.mean(
        torch.multiply(
            torch.exp(torch.multiply(5, y_true)),
            torch.square(torch.subtract(y_pred, y_true)),
        )
    )

    # Penalty for negative predictions
    negative_penalty = torch.mean(torch.relu(-y_pred))  # Penalizes when y_pred < 0

    # Adjust the weight of the penalty as needed (e.g., 10.0 can be modified)
    total_loss = base_loss + 10 * negative_penalty

    return total_loss
