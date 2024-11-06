import torch


def avg_loss(
    y: torch.Tensor, naive_predictor: callable, loss_function: callable
) -> float:
    """Calculate average loss for regression function."""
    total_loss = 0.0
    n_samples = len(y)
    for i in range(n_samples):
        y_pred = naive_predictor(y, i)
        total_loss += loss_function(y_pred, y[i]).item()
    return total_loss / n_samples


def lag24_baseline(
    loss_func: callable,
    y_test: torch.Tensor,
    model_loss: float,
    n_lag: int = 24,
) -> float:
    """Calculate the percentage loss of the lag24 baseline."""

    def naive_24h_predictor(y: torch.Tensor, idx: int) -> torch.Tensor:
        """Predict the value of y at index idx using the lag24 baseline."""
        return y[idx - n_lag]

    return avg_loss(y_test, naive_24h_predictor, loss_func) / model_loss
