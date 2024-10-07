import pandas as pd
from core.util.io import get_project_root, write_csv

# Define the JSON filename
root = get_project_root()
json_filename = root / "core/data/raw/caltech_ev_sessions.json"
output_dir = root / "core/data/processed/"

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

print(ev_sessions_df)

# Write the DataFrame to a CSV file
write_csv(ev_sessions_df, output_dir / "caltech_ev_sessions.csv")

print("Data has been written.")
