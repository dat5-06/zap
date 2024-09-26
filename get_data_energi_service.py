import pandas as pd
from pathlib import Path
from call_energi_service_api import call_energi_service_api


def get_data_energi_service() -> None:
    """Get data from the Energi Service API and save it in CSV format."""
    # Directory to save files
    output_dir = "energi_service_data"

    # dataset_name = "Forecasts_hour"
    # dataset_name = "ConsumptionCoverageLocationBased"
    start_date = "2022-01-01T00:00"
    end_date = "2023-12-31T00:00"
    sort = "HourUTC ASC"
    offset = 0
    filter_area = '{"PriceArea":["DK1"]}'
    connected_area = "DK1"

    # Get Forecast Data
    forecast_data = call_energi_service_api(
        "Forecasts_hour", start_date, end_date, offset, filter_area, sort
    )
    forecast_df = pd.DataFrame(forecast_data["records"])
    forecast_data = forecast_df.drop(
        columns=["HourUTC", "TimestampUTC", "TimestampDK", "PriceArea"]
    )
    forecast_df.to_csv(Path(output_dir) / "ForeCast1hour.csv", index=False)

    # Get Consumption Data for DK1
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
    consumption_data = consumption_df.drop(columns=["HourUTC", "PriceArea", "Updated"])
    consumption_df.to_csv(Path(output_dir) / "ConsumptionDK1.csv", index=False)


if __name__ == "__main__":
    get_data_energi_service()
