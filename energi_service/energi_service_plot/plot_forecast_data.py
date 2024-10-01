import matplotlib.pyplot as plt
import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame."""
    return pd.read_csv(path)


def plot_forecast_data(file_path: str, year: int, month: int) -> None:
    """Plot forecast data from a specified CSV file for a specific month."""
    data = read_csv(file_path)
    data["HourDK"] = pd.to_datetime(data["HourDK"])

    data_filtered = data[
        (data["HourDK"].dt.year == year) & (data["HourDK"].dt.month == month)
    ]

    data_filtered = data_filtered.set_index("HourDK")
    data_resampled = (
        data_filtered.groupby("ForecastType").resample("D").mean().reset_index()
    )

    plt.figure(figsize=(16, 8))
    forecast_types = data_resampled["ForecastType"].unique()

    for forecast_type in forecast_types:
        subset = data_resampled[data_resampled["ForecastType"] == forecast_type]

        plt.plot(
            subset["HourDK"],
            subset["Forecast1Hour"],
            label=f"{forecast_type} - 1 Hour Forecast",
            linestyle="-",
            linewidth=4,
            alpha=0.9,
        )

        plt.plot(
            subset["HourDK"],
            subset["ForecastCurrent"],
            label=f"{forecast_type} - Current Forecast",
            linestyle="--",
            linewidth=4,
            alpha=0.9,
        )

    plt.title(f"Forecast Data for {year}-{month}: 1 Hour and Current", fontsize=16)
    plt.xlabel("HourDK", fontsize=20)
    plt.ylabel("Forecast Value", fontsize=20)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

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
    plt.tight_layout()
    plt.savefig(f"forecast_plot_{year}_{month}.pdf")


plot_forecast_data(
    "../energi_service_data/ForeCast1hour_Onshore_Wind.csv", year=2023, month=5
)
