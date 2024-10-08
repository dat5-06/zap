from core.util.io import read_csv, write_csv, save_fig, read_xlsx
import pandas as pd


def park() -> None:
    """Process public charging station data."""
    original = read_xlsx("raw/trefor_park.xlsx")

    # crate deep copy that we can manupulate
    data = original.copy(deep=True)
    data = data.drop(columns=["Dato", "Time"])

    # remove values until first non-zero
    cleaned_data = data.apply(lambda col: col.iloc[col.ne(0).idxmax() :])

    # merge back together and save
    combined = pd.concat([original[["Dato", "Time"]], cleaned_data], axis=1)
    write_csv(combined, "interim/trefor_park.csv")


def household() -> None:
    """Preprocess the Trefor household data."""
    # Loading the original data into a pandas dataframe
    file_path = "raw/trefor_raw.csv"
    consumption = read_csv(file_path)

    # Dropping date and time, as this is not needed when cleaning
    df_original = consumption.drop(columns=["Dato", "Time"])

    # Copy is disconnected from the orignal df
    df_cleaning = df_original.copy(deep=True)

    # Only take valuse that are greater than 0
    non_zeroes = df_cleaning.apply(lambda series: series[series > 0])
    # And only take values that are less than 100
    # (some of it is weird, according to Trefor)
    cleaned = non_zeroes.apply(lambda series: series[series < 100])

    # Print the removed datapoints
    # datapoint_count = df_original.count().sum()
    # print(f"Original datapoints: {datapoint_count}")
    # print(f"Indicies removed: {datapoint_count - cleaned.count().sum()}")

    # remove households with maximum consumption less than 2kWh
    cleaned = cleaned.loc[:, (cleaned.max(axis=0) > 2)]
    # remove households with mean consumption less than 0.1kWh
    cleaned = cleaned.loc[:, (cleaned.mean(axis=0) > 0.1)]

    # Reads the date and time before transfering to csv file
    cleaned = pd.concat([consumption[["Dato", "Time"]], cleaned], axis=1)

    # Return the cleaned data into another csv
    output_path = "interim/trefor_cleaned.csv"
    write_csv(cleaned, output_path)

    print("Processed Trefor household data")


def plot_household() -> None:
    """Plot the combined the Trefor household data."""
    cleaned = read_csv("raw/trefor_cleaned.csv")
    # get mean consumption per hour for each household
    grouped = cleaned.drop(["Dato"], axis=1).groupby("Time").mean()

    # take mean of all households
    grouped["mean"] = grouped.to_numpy().mean(axis=1)

    # plot it
    grouped["mean"].plot(
        kind="bar",
        xlabel="Time of Day",
        ylabel="Mean Consumption (kWh)",
        # title="Mean Consumption per Hour",
        figsize=(10, 6),
        # rot="horizontal",
    )
    fig_output_path = "trefor/average_household.pdf"
    save_fig(fig_output_path)
    # plt.show()


def trefor() -> None:
    """Preprocess all Trefor data."""
    park()
    household()


if __name__ == "__main__":
    trefor()
    # plot_household()
