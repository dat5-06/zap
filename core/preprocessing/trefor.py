from core.util.io import read_csv, write_csv, read_xlsx
import pandas as pd
import numpy as np

capacities = [800, 2500, 2000, 800, 900, 1300]


def park_preprocess_lin() -> None:
    """Process public charging station data."""
    original = read_csv("interim/trefor_park.csv")

    for i, capacity in enumerate(capacities, 1):
        # normalize based on capacity to get relative (%) values
        original[f"Ladepark {i}"] = original[f"Ladepark {i}"] / capacity

    write_csv(original, "interim/park_timeseries_lin.csv")


def park_cleaning() -> None:
    """Process public charging station data."""
    original = read_xlsx("raw/trefor_park.xlsx")

    # crate deep copy that we can manupulate
    data = original.copy(deep=True)
    data = data.drop(columns=["Dato", "Time"])

    # remove values until first non-zero
    cleaned_data = data.apply(lambda col: col.iloc[col.ne(0).idxmax() :])

    # merge back together and save
    combined = pd.concat([original[["Dato", "Time"]], cleaned_data], axis=1)
    combined = month_of_year(combined)
    combined = week_day(combined)
    combined = hour_of_day(combined)
    write_csv(combined, "interim/trefor_park.csv")


def park_preprocess() -> None:
    """Process public charginc station data."""
    original = read_csv("interim/park_timeseries_lin.csv")

    for park_num in range(1, 7):
        park = pd.DataFrame(
            original[
                [
                    "Dato",
                    "Time",
                    "Hour_x",
                    "Hour_y",
                    "Day_x",
                    "Day_y",
                    "Month_x",
                    "Month_y",
                    f"Ladepark {park_num}",
                ]
            ].dropna()
        )
        park_renamed = park.rename(
            columns={f"Ladepark {park_num}": "Consumption", "Dato": "Date"}
        )
        write_csv(park_renamed, f"processed/park_{park_num}.csv")

    print("Processed Trefor park data")


def month_of_year(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column with the weekday of the date."""
    df["Dato"] = pd.to_datetime(df["Dato"], format="%d-%m-%Y")
    df["Month"] = df["Dato"].dt.month / 12
    df["Month_x"] = np.sin(2 * np.pi * df.Month)
    df["Month_y"] = np.cos(2 * np.pi * df.Month)
    return df


def week_day(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column with the weekday of the date."""
    df["Dato"] = pd.to_datetime(df["Dato"], format="%d-%m-%Y")
    df["Weekday"] = df["Dato"].dt.dayofweek / 7
    df["Day_x"] = np.sin(2 * np.pi * df.Weekday)
    df["Day_y"] = np.cos(2 * np.pi * df.Weekday)
    return df


def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column with the hour of the day."""
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M")
    df["Hour"] = df["Time"].dt.hour / 24
    df["Hour_x"] = np.sin(2 * np.pi * df.Hour)
    df["Hour_y"] = np.cos(2 * np.pi * df.Hour)
    return df


def trefor() -> None:
    """Preprocess all Trefor data."""
    park_cleaning()
    park_preprocess_lin()
    park_preprocess()
    print("Processed Trefor data")


if __name__ == "__main__":
    trefor()
