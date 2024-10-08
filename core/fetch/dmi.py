from sys import stderr

import pandas as pd
import requests
from tqdm import tqdm

from core.util.env import DMI_API_KEY
from core.util.io import get_data_root, write_csv


def _fetch_stations() -> None:
    """Fetch stations from DMI.

    NOTE: This is only used to get stations in the area.
    It does not actually save anything on disk.
    """
    # bbox is Treknatsområdet
    url = f"https://dmigw.govcloud.dk/v2/metObs/collections/station/items?bbox=8.1828,55.286,10.8415,55.8019&api-key={DMI_API_KEY}"

    data = requests.get(url, timeout=3).json()

    # Check if there are features returned in the response
    if "features" not in data:
        print("ERROR: Failed to fetch stations in Triangle Region", file=stderr)
        return
    print("Fetching station codes from DMI")

    stations = set([station["properties"]["stationId"] for station in data["features"]])

    base_url = "https://dmigw.govcloud.dk/v2/metObs/collections/observation/items"
    datetime = "2022-01-01T00:00:00Z/2024-01-01T00:00:00Z"
    limit = "200000"

    usable_stations = []

    for st in tqdm(stations):
        url = (
            f"{base_url}"
            f"?datetime={datetime}"
            f"&limit={limit}"
            "&parameterId=weather"
            f"&stationId={st}"
            f"&api-key={DMI_API_KEY}"
        )

        result = requests.get(url, timeout=10)
        res_data = result.json()
        if "features" not in res_data:
            continue

        for item in res_data["features"]:
            station = [item["properties"]["stationId"], res_data["numberReturned"]]
            if station not in usable_stations:
                usable_stations.append(station)

    print(f"Found {len(usable_stations)} usable stations in Triangle Region:")
    print(usable_stations)


# station id "06104" is Billund lufthavn
# station id "06080" is Esbjerg lufthavn
def _fetch_wcodes(force: bool = False) -> None:
    """Fetch weather codes from DMI.

    Arguments:
    ---------
        force: re-fetch data even if it exists

    """
    data_root = get_data_root()
    path = data_root / "external/w_d_codes.csv"
    if path.exists() and not force:
        print("DMI data already exists, and force was not supplied - skipping.")
        return
    print("Fetching weather codes from DMI")

    base_url = "https://dmigw.govcloud.dk/v2/metObs/collections/observation/items"
    datetime = "2022-01-01T00:00:00Z/2024-01-01T00:00:00Z"
    limit = "18000"
    station = "06104"

    # Parameters
    params = {
        "temp_mean_past1h": "temp",
        "wind_speed_past1h": "wind",
        "precip_past1h": "rain",
        "humidity_past1h": "humidity",
    }

    combined_data: pd.DataFrame = pd.DataFrame()

    for param, name in tqdm(params.items()):
        url = (
            f"{base_url}"
            f"?datetime={datetime}"
            f"&limit={limit}"
            f"&stationId={station}"
            f"&parameterId={param}"
            f"&api-key={DMI_API_KEY}"
        )

        data = requests.get(url, timeout=30).json()

        # Check if there are features returned in the response
        if "features" not in data:
            print(
                f"ERROR: Failed to fetch weather code '{param}' in Triangle Region",
                file=stderr,
            )
            continue

        # Normalize json data to use for dataframe manipulation
        data_to_df: pd.DataFrame = pd.json_normalize(data["features"])

        # Add time and value as columns
        data_to_df = data_to_df[["properties.observed", "properties.value"]]
        data_to_df.columns = ["observed", f"{name}"]

        # TODO: check for missing values? Possibly a cleaning task
        # rows can be removed with: df.dropna(subset=["column_name"])

        # Initialize df with time and value for first parameter,
        # else merge data on time.
        if combined_data.empty:
            combined_data = data_to_df
        else:
            combined_data = combined_data.merge(data_to_df, how="outer", on="observed")

        # TODO: This way of gathering data yields dataframes that have a variable
        # length of rows.
        # This should be reconsidered.

    combined_data = combined_data.sort_values("observed")

    write_csv(combined_data, "external/w_d_codes.csv")


def fetch_dmi() -> None:
    """Fetch all relevant data from DMI."""
    # fetch_stations()
    _fetch_wcodes()


if __name__ == "__main__":
    fetch_dmi()
