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
