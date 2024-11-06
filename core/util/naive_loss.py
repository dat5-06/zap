import torch
from scipy import stats


def avg_loss_linear(
    x: torch.Tensor, y: torch.Tensor, regression: callable, loss_function: callable
) -> float:
    """Calculate average loss for regression function."""
    loss = 0
    for i in range(len(x)):
        loss += loss_function(y[i], regression(x[i]))
    return loss / len(x)


def naive_linear_baseline(
    loss_func: callable,
    y_test: torch.Tensor,
    x_test: torch.Tensor,
    model_loss: float,
) -> float:
    """Find percentage difference between naive model loss and the inputted loss."""
    slope, intercept, _, _, _ = stats.linregress(x_test.flatten(), x_test.flatten())

    def regression(x: float) -> float:
        """Make a function for the regression."""
        return slope * x + intercept

    return avg_loss_linear(x_test, y_test, regression, loss_func) / model_loss
