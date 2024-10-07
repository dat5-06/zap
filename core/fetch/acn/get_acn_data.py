import requests
import json
from core.util.env import ACN_API_KEY
from core.util.io import get_project_root

# Create JSON file if it does not exist
rootpath = get_project_root()
output_file = rootpath / "core/data/raw/caltech_ev_sessions.json"
output_file.parent.mkdir(parents=True, exist_ok=True)

if not output_file.exists():
    with output_file.open("w") as file:
        json.dump([], file)

# Set the base URL and initial parameters
base_url = "https://ev.caltech.edu/api/v1/sessions/caltech"
params = {
    "where": (
        'connectionTime>="Wed, 1 May 2019 00:00:00 GMT" and '
        'connectionTime<="Thu, 2 May 2024 00:00:00 GMT"'
    ),
    "page": 1,
}
auth_token = ACN_API_KEY
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {ACN_API_KEY}",
}

# Open the file and load existing data
with output_file.open("r") as file:
    existing_data = json.load(file)

while True:
    response = requests.get(base_url, params=params, headers=headers, timeout=10)
    if response.status_code != 200:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        break

    data = response.json()
    new_data = data["_items"]

    # Append new data to the existing data and save to file
    existing_data.extend(new_data)
    with output_file.open("w") as file:
        json.dump(existing_data, file, indent=4)

    if "next" in data["_links"]:
        params["page"] += 1
        print(
            f"Fetching page {params['page']} this is slow, go to https://ev.caltech.edu/dataset"
        )
    else:
        print("All pages fetched.")
        break

print(f"Data successfully saved to {output_file}.")
