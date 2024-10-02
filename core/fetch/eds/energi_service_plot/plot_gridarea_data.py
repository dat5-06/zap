import matplotlib.pyplot as plt
import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame."""
    return pd.read_csv(path)


def plot_consumption_data(file_path: str, year: int, month: int) -> None:
    """Plot consumption data from a specified CSV file for a date range."""
    data = read_csv(file_path)
    data["HourDK"] = pd.to_datetime(data["HourDK"])

    # Filter data for the specified date range
    data_filtered = data[
        (data["HourDK"].dt.year == year) & (data["HourDK"].dt.month == month)
    ]

    # Set 'HourDK' as index
    data_filtered = data_filtered.set_index("HourDK")

    # Resample data to daily sum
    data_resampled = data_filtered.resample("D").sum().reset_index()

    # Define the columns to be plotted
    columns_to_plot = {
        "FlexSettledConsumption": "Flex Settled Consumption (MWh)",
        "HourlySettledConsumption": "Hourly Settled Consumption (MWh)",
    }

    colors = {"FlexSettledConsumption": "blue", "HourlySettledConsumption": "orange"}

    # Plot each column individually
    for column, ylabel in columns_to_plot.items():
        plt.figure(figsize=(16, 8))
        plt.plot(
            data_resampled["HourDK"],
            data_resampled[column],
            label=ylabel,
            linestyle="-",
            linewidth=4,
            color=colors[column],
            alpha=0.9,
        )
        plt.title(f"{ylabel} Data for {year}-{month}", fontsize=16)
        plt.xlabel("Date", fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.xticks(
            pd.date_range(
                start=data_resampled["HourDK"].min(),
                end=data_resampled["HourDK"].max(),
                freq="2D",
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

    # Plot both columns together
    plt.figure(figsize=(16, 8))
    for column, ylabel in columns_to_plot.items():
        plt.plot(
            data_resampled["HourDK"],
            data_resampled[column],
            label=ylabel,
            linestyle="-",
            linewidth=4,
            color=colors[column],
            alpha=0.9,
        )
    plt.title(f"Combined Consumption Data for {year}-{month}", fontsize=16)
    plt.xlabel("Date", fontsize=20)
    plt.ylabel("Consumption (MWh)", fontsize=20)
    plt.xticks(
        pd.date_range(
            start=data_resampled["HourDK"].min(),
            end=data_resampled["HourDK"].max(),
            freq="2D",
        ),
        rotation=45,
        fontsize=12,
    )
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=14, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"Combined_Consumption_plot_{year}_{month}.pdf")
    plt.show()


plot_consumption_data(
    "data/external/ConsumptionPerGridArea.csv",
    year=2023,
    month=5,
)
