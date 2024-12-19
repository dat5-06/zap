import pandas as pd
from core.util.io import get_data_root, write_csv
from datetime import datetime, timedelta
from core.preprocessing.trefor import month_of_year, week_day, hour_of_day
from tqdm import tqdm


def acn() -> None:
    """Convert Caltech EV sessions from JSON to CSV."""
    # External file should only have data from 1st December 2018 to 1st Jan 2019
    # As this is the test split they used in the paper
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


def calculate_overlap(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    timestep_start: datetime,
    timestep_end: datetime,
) -> float:
    """Calculate the overlap between a session and a timestep."""
    overlap_start = max(start_time, timestep_start)
    overlap_end = min(end_time, timestep_end)
    if overlap_start < overlap_end:
        overlap_duration = (overlap_end - overlap_start).total_seconds()
        session_duration = (end_time - start_time).total_seconds()
        return overlap_duration / session_duration
    return 0.0


def process_ev_data() -> None:
    """Convert Caltech EV sessions to hourly consumption data as Trefor."""
    root = get_data_root()
    input_filename = root / "interim/caltech_ev_sessions.csv"
    output_filename = root / "processed/caltech_ev_sessions.csv"
    ev_session_df = pd.read_csv(input_filename, sep=";")

    # Convertion to the datetime we have for trefor park
    ev_session_df["connectionTime"] = pd.to_datetime(
        ev_session_df["connectionTime"], format="%a, %d %b %Y %H:%M:%S GMT"
    )
    ev_session_df["disconnectTime"] = pd.to_datetime(
        ev_session_df["disconnectTime"], format="%a, %d %b %Y %H:%M:%S GMT"
    )

    # Generate hourly timesteps from Jan 1, 2019 00:00 to Dec 31, 2019 00:00
    timestep_start = datetime(2018, 4, 25, 11, 0)
    timestep_end = datetime(2021, 9, 14, 4, 0)
    timesteps = pd.date_range(timestep_start, timestep_end, freq="h")

    # Initialize a list to store results
    results = []

    # Iterate over each timestep
    for timestep in tqdm(timesteps):
        timestep_start = timestep
        timestep_end = timestep + timedelta(hours=1)

        # Initialize aggregated consumption for this timestep
        aggregated_consumption = 0

        # Iterate over each session
        for _, row in ev_session_df.iterrows():
            session_start = row["connectionTime"] - timedelta(
                hours=8
            )  # covert from GMT to PST
            session_end = row["disconnectTime"] - timedelta(
                hours=8
            )  # covert from GMT to PST
            kwh = float(row["kWhDelivered"].replace(",", "."))

            # Calculate the overlap between the session and the current timestep
            overlap_percentage = calculate_overlap(
                session_start, session_end, timestep_start, timestep_end
            )

            # Add the consumption contribution from this session
            aggregated_consumption += kwh * overlap_percentage

        # Add the results for this timestep to the list
        results.append(
            {
                "Dato": timestep.strftime("%Y-%m-%d"),
                "Time": timestep.strftime("%H:%M:%S"),
                "Consumption": aggregated_consumption,
            }
        )

    # Create a DataFrame for the results
    caltech_to_trefor_results_df = pd.DataFrame(results)

    # Add the Hour_x, Hour_y, Day_x, Day_y, Month_x, Month_y columns
    caltech_to_trefor_results_df = month_of_year(
        caltech_to_trefor_results_df, "%Y-%m-%d"
    )
    caltech_to_trefor_results_df = week_day(caltech_to_trefor_results_df, "%Y-%m-%d")
    caltech_to_trefor_results_df = hour_of_day(caltech_to_trefor_results_df, "%H:%M:%S")

    # Reorder the columns to ensure consumption is the last column
    caltech_to_trefor_results_df = caltech_to_trefor_results_df[
        [
            "Dato",
            "Time",
            "Hour_x",
            "Hour_y",
            "Day_x",
            "Day_y",
            "Month_x",
            "Month_y",
            "Consumption",
        ]
    ]

    write_csv(caltech_to_trefor_results_df, output_filename)
    print(f"Processed data saved to {output_filename}")


if __name__ == "__main__":
    acn()
    process_ev_data()
