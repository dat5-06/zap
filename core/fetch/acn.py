from math import ceil
import requests
import json

from tqdm import tqdm
from core.util.env import ACN_API_KEY
from core.util.io import get_data_root


def fetch_acn(force: bool = False) -> None:
    """Fetch data from ACN API.

    Arguments:
    ---------
        force: If True, overwrite existing file.

    """
    # Create JSON file if it does not exist
    rootpath = get_data_root()
    output_file = rootpath / "external/caltech_ev_sessions.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not force:
        print("ACN data already exists, and force was not supplied - skipping.")
        return
    print("Fetching data from ACN API")

    # Set the base URL and initial parameters
    base_url = "https://ev.caltech.edu/api/v1/sessions/caltech"
    params = {
        "where": (
            'connectionTime>="Wed, 1 Jan 2015 00:00:00 GMT" and '
            'connectionTime<="Thu, 10 Dec 2024 00:00:00 GMT"'
        ),
        "page": 1,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer b1pkX7JrolrALd-UyWSR3-TvwL678PwHgs-zpy3_gLA",
    }

    initial_req = requests.get(base_url, params=params, headers=headers, timeout=10)
    meta = initial_req.json()["_meta"]
    page_count = ceil(meta["total"] / meta["max_results"])

    combined_data = []

    for _ in tqdm(range(page_count)):
        response = requests.get(base_url, params=params, headers=headers, timeout=10)

        if response.status_code != 200:
            print(f"ERROR: Failed to fetch data. Status code: {response.status_code}")
            break

        data = response.json()
        combined_data.extend(data["_items"])

        if "next" not in data["_links"]:
            break
        params["page"] += 1

    with output_file.open("w") as file:
        json.dump(combined_data, file, indent=4)

    print(f"ACN Data successfully written to {output_file}.")


if __name__ == "__main__":
    fetch_acn(force=True)
