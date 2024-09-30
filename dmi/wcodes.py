import os
from pathlib import Path
import requests
import csv

from dotenv import load_dotenv

load_dotenv()
DMI_API_KEY = os.getenv("DMI_API_KEY")

# from util.env import DMI_API_KEY

# I use dis to fetch the data from the API and generate the noice csv files
# For the first part we use station id "06104", which is Billund lufthavn
# For the second part we use station id "06080", which is Esbjerg lufthavn

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

# Create empty dictionaries to store observations for each parameter
observations = {"temp": {}, "wind": {}, "rain": {}, "humidity": {}}


# Fetch and store data for each parameter
for param, name in params.items():
    url = (
        f"{base_url}"
        f"?datetime={datetime}"
        f"&limit={limit}"
        f"&stationId={station}"
        f"&parameterId={param}"
        f"&api-key={DMI_API_KEY}"
    )
    # Sometimes the program doesn't work with timeout
    # but ruff insists, so just delete if no work
    response = requests.get(url, timeout=30)
    data = response.json()

    # Store observations by timestamp
    for item in data.get("features", []):
        observed = item["properties"].get("observed")
        value = item["properties"].get("value")
        if value is not None:
            observations[name][observed] = value


# Open a CSV file for writing
with Path("w_d_codes.csv").open(mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(
        [
            "Observed",
            "Temperature",
            "Wind Speed",
            "Rainfall",
            "Humidity",
        ]
    )

    # Get all unique timestamps
    all_timestamps = (
        set(observations["temp"].keys())
        | set(observations["wind"].keys())
        | set(observations["rain"].keys())
        | set(observations["humidity"].keys())
    )

    # Write data row by row
    for timestamp in sorted(all_timestamps):
        temp = observations["temp"].get(timestamp, "")
        wind = observations["wind"].get(timestamp, "")
        rain = observations["rain"].get(timestamp, "")
        humidity = observations["humidity"].get(timestamp, "")
        writer.writerow([timestamp, temp, wind, rain, humidity])
