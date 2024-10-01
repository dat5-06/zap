import requests


def call_energi_service_api(
    dataset_name: str,
    start_date: str,
    end_date: str,
    offset: str,
    filter_area: str,
    sort: str,
    connected_area: str = None,
    grid_company: str = None,
) -> dict:
    """Call the Energi Service API and return the response in JSON format.

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
