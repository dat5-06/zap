import matplotlib.pyplot as plt
import pandas as pd
from util.util import read_csv


def plot_consumption_data(file_path: str, year: int, month: int) -> None:
    """Plot consumption data from a specified CSV file for a date range."""
    data = read_csv(file_path)
    data["HourDK"] = pd.to_datetime(data["HourDK"])

    # Filter data for the specified date range
    data_filtered = data[
        (data["HourDK"].dt.year == year) & (data["HourDK"].dt.month == month)
    ]

    if data_filtered.empty:
        print(f"No data available for {year}-{month}.")
        return

    # Set 'HourDK' as index
    data_filtered = data_filtered.set_index("HourDK")

    # Resample data to daily sum
    data_resampled = data_filtered.resample("D").sum().reset_index()

    # Define the columns to be plotted
    columns_to_plot = {"ShareMWh": "Consumption(MWh)", "SharePPM": "Share PPM(MWh)"}

    for column, ylabel in columns_to_plot.items():
        # Create a figure for each column
        plt.figure(figsize=(16, 8))

        # Plot the data for the specific column
        plt.plot(
            data_resampled["HourDK"],
            data_resampled[column],
            label=ylabel,
            linestyle="-",
            linewidth=4,
            color="blue" if column == "ShareMWh" else "orange",
            alpha=0.9,
        )

        # Set plot title and labels
        plt.title(f"{ylabel} Data for {year}-{month}", fontsize=16)
        plt.xlabel("Date", fontsize=20)
        plt.ylabel(ylabel, fontsize=20)

        # Set x-ticks to every other day
        plt.xticks(
            pd.date_range(
                start=data_resampled["HourDK"].min(),
                end=data_resampled["HourDK"].max(),
                freq="2D",  # Set to every other day
            ),
            rotation=45,
            fontsize=12,
        )

        plt.yticks(fontsize=20)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=14, loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{column}_plot_{year}_{month}.pdf")
        plt.show()


plot_consumption_data(
    "data/external/ConsumptionDK1.csv",
    year=2023,
    month=5,
)
