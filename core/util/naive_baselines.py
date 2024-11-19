import torch
from scipy import stats


def _24hlnbe(
    loss_function: callable,
    y_test: torch.Tensor,
    model_loss: float,
    n_lag: int = 24,
) -> float:
    """Calculate the percentage loss of the 24 hour lag naive baseline."""
    return loss_function(y_test[:-n_lag], y_test[n_lag:]) / model_loss


def nbe(
    loss_function: callable,
    y_test: torch.Tensor,
    x_test: torch.Tensor,
    model_loss: float,
) -> float:
    """Find percentage difference between naive model loss and the inputted loss."""
    slope, intercept, _, _, _ = stats.linregress(x_test.flatten(), x_test.flatten())

    def regression(x: torch.tensor) -> torch.tensor:
        """Make a function for the regression."""
        for i in range(len(x)):
            x[i] = slope * x[i] + intercept
        return x

    return loss_function(y_test, regression(x_test)) / model_loss
