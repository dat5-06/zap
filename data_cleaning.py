import pandas as pd

# Loading the orignal data into a pandas dataframe
file_path = "trefor.csv"
df = pd.read_csv(file_path)

# Dropping date and time, as this is not needed when cleaning
df_original = df.drop(columns=["Dato", "Time"])
datapoint_count = df_original.count().sum()

# Copy is disconnected from the orignal df
df_cleaning = df_original.copy(deep=True)

# Only take valuse that are greater than 0
non_zeroes = df_cleaning.apply(lambda series: series[series > 0])
# And only take values that are less than 100
cleaned = non_zeroes.apply(lambda series: series[series < 100])

# Print the removed datapoints
print(f"Original datapoints: {datapoint_count}")
print(f"Indicies removed: {datapoint_count - cleaned.count().sum()}")

# Reads the date and time before transfering to csv file
cleaned = pd.concat([df[["Dato", "Time"]], cleaned], axis=1)

# Return the cleaned data into another csv
cleaned.to_csv("trefor_cleaned.csv", sep=";", decimal=",", index=False)
