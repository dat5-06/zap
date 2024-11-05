import matplotlib.pyplot as plt
import torch


def plot_predictions(
    start_day: int, end_day: int, y_test: list, predicted: list
) -> None:
    """Plot the model predictions and the actual values from startDay to endDay."""
    first_column_actual = y_test[start_day * 24].flatten().to("cpu")
    first_column_predicted = predicted[start_day * 24].flatten().to("cpu")
    for i in range(start_day, end_day):
        first_column_actual = torch.cat(
            (first_column_actual, y_test[i * 24].flatten().to("cpu"))
        )
        first_column_predicted = torch.cat(
            (first_column_predicted, predicted[i * 24].flatten().to("cpu"))
        )
    plt.plot(first_column_actual, label="First Column Actual Consumption")
    plt.plot(first_column_predicted, label="First Column Predicted Consumption")
    plt.xlabel("Hour")
    plt.ylabel("Consumption")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Position the legend outside
    plt.show()
