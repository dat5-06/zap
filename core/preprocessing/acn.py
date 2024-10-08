import pandas as pd
from core.util.io import get_data_root, write_csv


def acn() -> None:
    """Convert Caltech EV sessions from JSON to CSV."""
    # Define the JSON filename
    root = get_data_root()
    json_filename = root / "external/caltech_ev_sessions.json"
    output_filename = "interim/caltech_ev_sessions.csv"

    # Read the JSON data into a pandas DataFrame
    ev_sessions_df = pd.read_json(json_filename)

    # Define the columns to include in the CSV
    columns = [
        "_id",
        "connectionTime",
        "disconnectTime",
        "kWhDelivered",
        "doneChargingTime",
    ]

    # Select only the desired columns
    ev_sessions_df = ev_sessions_df[columns]

    # Write the DataFrame to a CSV file
    write_csv(ev_sessions_df, output_filename)

    print(f"Saved Caltech EV sessions to {root / output_filename}")


if __name__ == "__main__":
    acn()
