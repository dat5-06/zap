import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_predictions(
    start_day: int, end_day: int, y_test: list, predicted: list
) -> None:
    """Plot the model predictions and the actual values from startDay to endDay."""
    first_column_actual = y_test[start_day * 24].flatten().to("cpu")
    first_column_predicted = predicted[start_day * 24].flatten().to("cpu")
    for i in range(start_day + 1, end_day):
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


def plot_every_model(
    start_day: int,
    end_day: int,
    y_test: list,
    predicted_lstm: torch.Tensor,
    predicted_gru: torch.Tensor,
    predicted_cnn_lstm: torch.Tensor,
    outfile: str | None = None,
    hour_offset: int = 0,
) -> None:
    """Plot all models compared to the ground truth."""

    def index(d: int) -> int:
        return d * 24 + hour_offset

    ground_truth = y_test[index(start_day)].flatten().to("cpu")
    lstm = predicted_lstm[index(start_day)].flatten().to("cpu")
    gru = predicted_gru[index(start_day)].flatten().to("cpu")
    cnn_lstm = predicted_cnn_lstm[index(start_day)].flatten().to("cpu")

    for i in range(start_day + 1, end_day):
        ground_truth = torch.cat((ground_truth, y_test[index(i)].flatten().to("cpu")))
        lstm = torch.cat((lstm, predicted_lstm[index(i)].flatten().to("cpu")))
        gru = torch.cat((gru, predicted_gru[index(i)].flatten().to("cpu")))
        cnn_lstm = torch.cat(
            (cnn_lstm, predicted_cnn_lstm[index(i)].flatten().to("cpu"))
        )

    x = np.arange((end_day - start_day) * 24) + hour_offset

    plt.plot(x, ground_truth, label="Ground Truth")
    plt.plot(x, lstm, label="LSTM", linewidth=1)
    plt.plot(x, gru, label="GRU", linewidth=1)
    plt.plot(x, cnn_lstm, label="CNN-LSTM", linewidth=1)
    for day in range(end_day - start_day):
        plt.axvline(
            x=24 * (day + 1),
            ls=":",
            ymin=0.05,
            ymax=0.95,
            color="gray",
        )

    plt.axhline(y=0, color="grey")

    plt.xticks(
        np.arange(0 + hour_offset, (end_day - start_day) * 24 + hour_offset + 1, 3)
    )

    plt.xlabel("Hour")
    plt.ylabel("Relative Consumption")
    plt.legend()
    if outfile:
        plt.savefig(outfile)
    plt.show()
