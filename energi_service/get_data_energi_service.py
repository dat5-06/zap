import pandas as pd
from pathlib import Path
from call_energi_service_api import call_energi_service_api


def create_output_directory(directory: str) -> None:
    """Create a directory if it does not exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_dataframe_to_csv(df: pd.DataFrame, filepath: str) -> None:
    """Save a DataFrame to a CSV file."""
    df.to_csv(filepath, index=False)


def get_consumption_data(
    start_date: str,
    end_date: str,
    offset: int,
    filter_area: str,
    sort: str,
    output_dir: str,
) -> None:
    """Get and save consumption data for DK1."""
    connected_area = "DK1"
    consumption_data = call_energi_service_api(
        "ConsumptionCoverageLocationBased",
        start_date,
        end_date,
        offset,
        filter_area,
        sort,
        connected_area,
    )

    consumption_df = pd.DataFrame(consumption_data["records"])
    consumption_df = consumption_df.drop(columns=["HourUTC", "PriceArea", "Updated"])
    save_dataframe_to_csv(consumption_df, Path(output_dir) / "ConsumptionDK1.csv")


def get_forecast_data(
    forecast_type: str,
    start_date: str,
    end_date: str,
    offset: int,
    sort: str,
    output_dir: str,
) -> None:
    """Get and save forecast data for a given type."""
    filter_area = f'{{"PriceArea":["DK1"], "ForecastType":["{forecast_type}"]}}'
    forecast_data = call_energi_service_api(
        "Forecasts_hour", start_date, end_date, offset, filter_area, sort
    )

    forecast_df = pd.DataFrame(forecast_data["records"])
    forecast_df = forecast_df.drop(
        columns=["HourUTC", "TimestampUTC", "TimestampDK", "PriceArea"]
    )
    output_file = (
        Path(output_dir) / f"ForeCast1hour_{forecast_type.replace(' ', '_')}.csv"
    )
    save_dataframe_to_csv(forecast_df, output_file)


def get_grid_area_consumption(
    start_date: str, end_date: str, offset: int, sort: str, output_dir: str
) -> None:
    """Get and save consumption data for a specific grid area."""
    grid_area_filter = '{"GridCompany":["244"]}'
    grid_area_consumption = call_energi_service_api(
        "ConsumptionPerGridarea",
        start_date,
        end_date,
        offset,
        filter_area=grid_area_filter,
        sort=sort,
    )

    grid_area_consumption_df = pd.DataFrame(grid_area_consumption["records"])
    grid_area_consumption_df = grid_area_consumption_df.drop(
        columns=["HourUTC", "ResidualConsumption"]
    )
    save_dataframe_to_csv(
        grid_area_consumption_df, Path(output_dir) / "ConsumptionPerGridArea.csv"
    )


def get_data_energi_service() -> None:
    """Get data from the API and save it in CSV format."""
    output_dir = "energi_service_data"
    create_output_directory(output_dir)

    start_date = "2022-01-01T00:00"
    end_date = "2023-12-31T00:00"
    sort = "HourUTC ASC"
    offset = 0
    filter_area = '{"PriceArea":["DK1"]}'
    forecast_types = ["Solar", "Onshore Wind", "Offshore Wind"]

    # Get consumption data for DK1
    get_consumption_data(start_date, end_date, offset, filter_area, sort, output_dir)

    # Get forecast data for each type
    for forecast_type in forecast_types:
        get_forecast_data(forecast_type, start_date, end_date, offset, sort, output_dir)

    # Get grid area consumption data
    get_grid_area_consumption(start_date, end_date, offset, sort, output_dir)


if __name__ == "__main__":
    get_data_energi_service()
