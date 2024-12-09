import math


def mae(y: list, y_hat: list) -> float:
    """Calculate MAE."""
    sum_tracker = 0
    for i in range(len(y)):
        sum_tracker += abs(y[i] - y_hat[i])
    return sum_tracker / len(y)


def rmse(y: list, y_hat: list) -> float:
    """Calculate RMSE."""
    sum_tracker = 0
    for i in range(len(y)):
        sum_tracker += math.pow(y[i] - y_hat[i], 2)
    return math.sqrt(sum_tracker / len(y))


def smape(y: list, y_hat: list) -> float:
    """Calculate sMAPE."""
    sum_tracker = 0
    for i in range(len(y)):
        if abs(y[i]) + abs(y_hat[i]) != 0:
            sum_tracker += abs(y[i] - y_hat[i]) / ((abs(y[i]) + abs(y_hat[i])) / 2)
    return (100 / len(y)) * sum_tracker


def adjusted_smape(y: list, y_hat: list, margin: float = 0.05) -> float:
    """Calculate sMAPE on all values in y that are above margin."""
    indicies = [i for i, pred in enumerate(y) if pred > margin]
    y_above_threshold = [y[i] for i in indicies]
    hat_above_threshold = [y_hat[i] for i in indicies]

    return smape(y_above_threshold, hat_above_threshold)
