from core.util.io import read_csv, write_csv, read_xlsx
import pandas as pd


def households() -> None:
    """Process household data."""
    file_path = "core/data/interim/trefor_cleaned.csv"
    trefor_data = read_csv(file_path)

    # TODO: Date and time column manipulation are uncommented, can be added
    # later to include as features, otherwise delete. Currently the output is
    # total consumption and a lookback window specified with n_steps
    #
    # Merge date and time columns into one and convert str to datetime
    # trefor_data["Date_Time"] = pd.to_datetime(
    #     trefor_data.pop("Dato") + " " + trefor_data.pop("Time"),
    #     format="%d-%m-%Y %H:%M"
    # )
    # trefor_data.insert(0, "Date_Time", trefor_data.pop("Date_Time"))
    trefor_data = trefor_data.drop(["Dato", "Time"], axis=1)

    # Create new row with total consumption
    # trefor_data["Total_Consumption"] = trefor_data[
    #     list(trefor_data.drop("Date_Time", axis=1))
    # ].sum(axis=1)
    trefor_data["Total_Consumption"] = trefor_data.sum(axis=1)
    trefor_data = trefor_data.filter(["Total_Consumption"], axis=1)

    # Sets the date time column as index and effectively ignores it.
    # trefor_data = trefor_data.filter(["Date_Time", "Total_Consumption"], axis=1)
    # trefor_data = trefor_data.set_index("Date_Time")

    # Sets number of lookback windows
    # TODO: This could be added to a configuration file for centralized variable
    # manipulation.
    n_steps = 5
    for i in range(1, n_steps + 1):
        trefor_data[f"Total_Consumption(t-{n_steps+1-i})"] = trefor_data[
            "Total_Consumption"
        ].shift(n_steps + 1 - i)

    # Drop rows or columns with missing data
    trefor_data = trefor_data.dropna()

    # Outputs the now processed data as csv. This should be torch.Dataset ready.
    output_path = "core/data/processed/trefor_final.csv"
    write_csv(trefor_data, output_path)


def public() -> None:
    """Process public charging station data."""
    original = read_xlsx("core/data/raw/trefor_public.xlsx")

    # crate deep copy that we can manupulate
    data = original.copy(deep=True)
    data = data.drop(columns=["Dato", "Time"])

    # remove values until first non-zero
    cleaned_data = data.apply(lambda col: col.iloc[col.ne(0).idxmax() :])

    # merge back together and save
    combined = pd.concat([original[["Dato", "Time"]], cleaned_data], axis=1)
    write_csv(combined, "core/data/interim/trefor_public.csv")


if __name__ == "__main__":
    households()
    public()
