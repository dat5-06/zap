import torch
import numpy as np


def naive(x_test: torch.Tensor) -> torch.Tensor:
    """Linear regression over the lookback window of 96."""
    # x-values for naive baseline regression
    # naive baseline is always 96 lookback window
    x = np.arange(96) - 96

    # iterate testset
    naive_prediction = []
    for item in x_test:
        # check the entire lookback window, but only get the consumption
        coef = np.polyfit(x, item[:, -1], 1)
        regression_fn = np.poly1d(coef)
        naive_prediction.append(regression_fn(np.arange(24)))

    return torch.tensor(np.array(naive_prediction))


def _24hlag(x_test: torch.Tensor) -> torch.Tensor:
    """Naive baseline predicting output to be equal to the last 24 hours."""
    return x_test[:, -24:, -1]
