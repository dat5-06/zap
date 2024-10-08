import requests
import pandas as pd
from core.util.io import write_csv


def _call_eds_api(
    dataset_name: str,
    start_date: str,
    end_date: str,
    offset: int,
    filter_area: str,
    sort: str,
    connected_area: str | None = None,
    grid_company: str | None = None,
) -> dict:
    """Call the Energi Data Service API and return the response in JSON format.

    This method constructs the URL with the given parameters and makes a GET request.
    """
    base_url = f"https://api.energidataservice.dk/dataset/{dataset_name}"
    params = {
        "offset": offset,
        "start": start_date,
        "end": end_date,
        "filter": filter_area,
        "sort": sort,
        "connected_area": connected_area,
        "GridCompany": grid_company,
    }

    # Construct the URL with the parameters
    url = f"{base_url}?" + "&".join(
        f"{key}={value}" for key, value in params.items() if value is not None
    )

    response = requests.get(url, timeout=10)

    return response.json()


def _consumption_data(
    filter_area: str,
    start_date: str,
    end_date: str,
    offset: int,
    sort: str,
) -> None:
    """Get and save consumption data for DK1."""
    print("Fetching consumption data for DK1, through EDS")

    connected_area = "DK1"
    consumption_data = _call_eds_api(
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
    write_csv(consumption_df, "external/ConsumptionDK1.csv")


def _forecast_data(
    forecast_type: str,
    start_date: str,
    end_date: str,
    offset: int,
    sort: str,
) -> None:
    """Get and save forecast data for a given type."""
    print(f"Fetching {forecast_type} forecast data for DK1, through EDS")

    filter_area = f'{{"PriceArea":["DK1"], "ForecastType":["{forecast_type}"]}}'
    forecast_data = _call_eds_api(
        "Forecasts_hour", start_date, end_date, offset, filter_area, sort
    )

    forecast_df = pd.DataFrame(forecast_data["records"])
    forecast_df = forecast_df.drop(
        columns=["HourUTC", "TimestampUTC", "TimestampDK", "PriceArea"]
    )
    write_csv(
        forecast_df, f"external/ForeCast1hour_{forecast_type.replace(' ', '_')}.csv"
    )


def _grid_area_consumption(
    start_date: str, end_date: str, offset: int, sort: str
) -> None:
    """Get and save consumption data for a specific grid area."""
    print("Fetching grid area consumption data, through EDS")

    grid_area_filter = '{"GridCompany":["244"]}'
    grid_area_consumption = _call_eds_api(
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
    write_csv(grid_area_consumption_df, "external/ConsumptionPerGridArea.csv")


def fetch_eds() -> None:
    """Get data from the API and save it in CSV format."""
    start_date = "2022-01-01T00:00"
    end_date = "2023-12-31T00:00"
    sort = "HourUTC ASC"
    offset = 0
    filter_area = '{"PriceArea":["DK1"]}'
    forecast_types = ["Solar", "Onshore Wind", "Offshore Wind"]

    # Get consumption data for DK1
    _consumption_data(filter_area, start_date, end_date, offset, sort)

    # Get forecast data for each type
    for forecast_type in forecast_types:
        _forecast_data(forecast_type, start_date, end_date, offset, sort)

    # Get grid area consumption data
    _grid_area_consumption(start_date, end_date, offset, sort)


if __name__ == "__main__":
    fetch_eds()
