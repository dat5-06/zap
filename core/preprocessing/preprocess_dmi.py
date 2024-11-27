from core.util.io import read_csv, write_csv
import pandas as pd


def remove_timeformatting() -> None:
    """Clean DMI data and save to disk."""
    original = read_csv("external/w_d_codes.csv")

    # Split 'observed' into 'Date' and 'Time'
    original["Date"] = original["observed"].str.split("T").str[0]
    time = original["observed"].str.split("T").str[1]

    # Clean 'Time': remove trailing "00Z" and adjust leading zeros
    time = time.str.replace("Z", "", regex=False)
    # Set 'Time' in original
    original["Time"] = time

    # Drop the original 'observed' column
    original = original.drop(columns=["observed"])

    # Save cleaned data
    write_csv(original, "interim/interim_dmi.csv")


def dmi_split_weathertype() -> None:
    """Split DMI data into separate files based on weather type."""
    weather_types = ["wind", "rain", "humidity", "temp"]
    original = read_csv("interim/interim_dmi.csv")
    date = original["Date"]
    time = original["Time"]

    for weather_type in weather_types:
        if weather_type in original.columns:
            weather_type_data = original[
                weather_type
            ]  # Combine date, time, and normalized weather type
            combined = pd.concat([date, time, weather_type_data], axis=1)
            combined.columns = ["Date", "Time", weather_type]
            write_csv(combined, f"processed/dmi_{weather_type}.csv")


def clean_dmi() -> None:
    """Clean and split DMI data."""
    remove_timeformatting()
    dmi_split_weathertype()


if __name__ == "__main__":
    clean_dmi()
