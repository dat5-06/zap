import requests
import pandas as pd
from core.util.io import write_csv
from core.util.env import DMI_API_KEY

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

combined_data = pd.DataFrame()

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

    # Normalize json data to use for dataframe manipulation
    data_to_df = pd.json_normalize(data["features"])

    # Add time and value as columns
    data_to_df = data_to_df[["properties.observed", "properties.value"]]
    data_to_df.columns = ["observed", f"{name}"]

    # TODO: check for missing values? Possibly a cleaning task
    # rows can be removed with: df.dropna(subset=["column_name"])

    # Initialize df with time and value for first parameter, else merge data on
    # time.
    if combined_data.empty:
        combined_data = data_to_df
    else:
        combined_data = combined_data.merge(data_to_df, how="outer", on="observed")

    # TODO: This way of gathering data yields dataframes that have variable row entries.
    # This should be reconsidered.

combined_data = combined_data.sort_values(by="observed", ascending=True)

combined_data = pd.DataFrame(combined_data)

output_path = "core/data/external/w_d_codes.csv"
write_csv(combined_data, output_path)
