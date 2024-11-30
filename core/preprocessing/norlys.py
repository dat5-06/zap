from core.util.io import write_csv, read_xlsx_sheet, read_csv
import numpy as np
import pandas as pd


def norlys() -> None:
    """Order of function calls in file."""
    norlys_split()
    norlys_add_features()


def norlys_split() -> None:
    """Split the data into private- and public chargning."""
    original = read_xlsx_sheet(
        "./raw/norlys_data.xlsx", "kwh_percentage", index_col=None
    )

    # Deep copy the original data and drop the specified columns
    data = original.copy(deep=True)

    # Split the data based on the "segment" column
    consumer_data = data[data["segment"] == "Consumer"].drop(columns=["segment"])
    public_data = data[data["segment"] == "Norlys Public"].drop(columns=["segment"])

    write_csv(consumer_data, "interim/norlys_cons_interim.csv")
    write_csv(public_data, "interim/norlys_pub_interim.csv")


def norlys_add_features() -> None:
    """Add column to datasets for total consumption with january as reference point."""
    original_cons = read_csv("interim/norlys_cons_interim.csv")
    original_pub = read_csv("interim/norlys_pub_interim.csv")

    cons_copy, pub_copy = get_indexes(
        original_cons=original_cons, original_pub=original_pub
    )

    cons_copy = add_seasonality(cons_copy)
    pub_copy = add_seasonality(pub_copy)

    write_csv(cons_copy, "processed/norlys/norlys_cons.csv")
    write_csv(pub_copy, "processed/norlys/norlys_pub.csv")


def get_indexes(
    original_cons: pd.DataFrame, original_pub: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add indexes as column to datasets."""
    indexes = read_xlsx_sheet("./raw/norlys_data.xlsx", sheet="index", index_col=0)

    cons_index_dict = indexes.loc["Consumer"].to_dict()
    pub_index_dict = indexes.loc["Norlys Public"].to_dict()

    cons_w_index = get_index_csv(original=original_cons, index_dict=cons_index_dict)

    pub_w_index = get_index_csv(original=original_pub, index_dict=pub_index_dict)

    return (cons_w_index, pub_w_index)


def get_index_csv(original: pd.DataFrame, index_dict: pd.DataFrame) -> pd.DataFrame:
    """Format the file to target file."""
    months_dict = {
        3: "mar",
        4: "apr",
        5: "maj",
        6: "jun",
        7: "jul",
        8: "aug",
        9: "sep",
        10: "okt",
    }

    copy = original.copy(deep=True)

    copy["month"] = pd.DatetimeIndex(copy["date"]).month
    copy["month_name"] = copy["month"].map(months_dict).drop(columns=["month"])

    # Map the index values to copy based on month_name
    copy["index"] = copy["month_name"].map(index_dict)

    copy = copy.drop(columns=["month", "month_name"])

    return copy


def add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column with the weekday of the date."""
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["weekday"] = df["date"].dt.dayofweek / 7
    df["day_x"] = np.sin(2 * np.pi * df.weekday)
    df["day_y"] = np.cos(2 * np.pi * df.weekday)
    df["month"] = df["date"].dt.month / 12
    df["month_x"] = np.sin(2 * np.pi * df.month)
    df["month_y"] = np.cos(2 * np.pi * df.month)
    data = df[[c for c in df if c not in ["kwh_percentage"]] + ["kwh_percentage"]]
    return data


if __name__ == "__main__":
    norlys()
