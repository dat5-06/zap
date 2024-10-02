import requests

from core.util.env import DMI_API_KEY

# I use dis to find the stations which is located within the bbox area,
# which is focused on the "trekansområde"


url = f"https://dmigw.govcloud.dk/v2/metObs/collections/station/items?bbox=8.1828,55.286,10.8415,55.8019&api-key={DMI_API_KEY}"

x = requests.get(url, timeout=3)

# Parse the JSON response
data = x.json()

stations = []


# Check if there are features returned in the response
if "features" in data:
    # Extract features and sort by the "name" property inside "properties"
    sorted_stations = sorted(
        data["features"], key=lambda station: station["properties"]["name"]
    )

    # Print out the "name" and "stationId" for each station
    for station in sorted_stations:
        name = station["properties"]["name"]
        station_id = station["properties"]["stationId"]
        print(f"Name: {name}, Station ID: {station_id}")
        if not (stations.__contains__(station_id)):
            stations.append(station_id)
else:
    print("No features found in the response")

print(stations)

base_url = "https://dmigw.govcloud.dk/v2/metObs/collections/observation/items"
datetime = "2022-01-01T00:00:00Z/2024-01-01T00:00:00Z"
limit = "200000"

usable_stations = []

for st in stations:
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
    if "features" in res_data:
        for item in res_data["features"]:
            if not usable_stations.__contains__(
                [item["properties"]["stationId"], res_data["numberReturned"]]
            ):
                usable_stations.append(
                    [item["properties"]["stationId"], res_data["numberReturned"]]
                )
    else:
        print("No stations, area sucks")

print(usable_stations)
