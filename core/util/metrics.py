import math


def mae(y: list, y_hat: list) -> float:
    """Calculate MAE."""
    sum_tracker = 0
    for i in range(len(y)):
        for j in range(len(y[i])):
            sum_tracker += abs(y[i][j] - y_hat[i][j])
    return float(sum_tracker / (len(y) * 24))


def rmse(y: list, y_hat: list) -> float:
    """Calculate RMSE."""
    sum_tracker = 0
    for i in range(len(y)):
        for j in range(len(y[i])):
            sum_tracker += math.pow(y[i][j] - y_hat[i][j], 2)
    return math.sqrt(sum_tracker / (len(y) * 24))


def smape(y: list, y_hat: list) -> float:
    """Calculate sMAPE."""
    sum_tracker = 0
    for i in range(len(y)):
        for j in range(len(y[i])):
            if abs(y[i][j]) + abs(y_hat[i][j]) != 0:
                sum_tracker += abs(y[i][j] - y_hat[i][j]) / (
                    (abs(y[i][j]) + abs(y_hat[i][j])) / 2
                )
    return float((100 / (len(y) * 24)) * sum_tracker)
