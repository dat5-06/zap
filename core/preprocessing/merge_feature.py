from core.util.io import read_csv, write_csv
import pandas as pd


def clean_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """Remove '1900-01-01' from the Time column and strip any extra spaces."""
    df["Time"] = df["Time"].str.replace("1900-01-01 ", "", regex=False).str.strip()
    return df


def combine_park_with_feature(feature_list: list) -> None:
    """Combine park data with features based on Date and Time."""
    for i in range(1, 7):
        park = read_csv(f"processed/park_{i}.csv")

        # Clean 'Time' column in park data
        park = clean_time_column(park)

        # Sequentially merge features
        for feature in feature_list:
            feature_file = read_csv(f"processed/{feature}.csv")

            # Clean 'Time' column in feature data
            feature_file = clean_time_column(feature_file)

            # Merge on 'Date' and 'Time'
            park = park.merge(feature_file, on=["Date", "Time"], how="inner")

        # Drop rows with any missing values
        park = park.dropna()

        # Write the final merged dataframe to CSV
        write_csv(park, f"merged/park_{i}.csv")


if __name__ == "__main__":
    combine_park_with_feature(["dmi_wind", "dmi_rain", "dmi_humidity", "dmi_temp"])
