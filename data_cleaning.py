import pandas as pd
import numpy as np
from data_cleaning_functions import find_first_non_zero, find_last_non_zero

# Loading the orignal data into a pandas dataframe
file_path = 'trefor.csv'
df = pd.read_csv(file_path)

# Dropping date and time, as this is not needed when cleaning
df_original = df.drop(columns=['Dato', 'Time'])

# Copy is disconnected from the orignal df
df_cleaning = df_original.copy(deep=True)

# For each column we find the first non zero and the last non zero and stores all indicies in a df
first_non_zero_idx = df_cleaning.apply(find_first_non_zero)
last_non_zero_idx = df_cleaning.apply(find_last_non_zero)

for column in df_cleaning.columns:
    # We find the indicies for this specific column
    first_non_zero = first_non_zero_idx[column]
    last_non_zero = last_non_zero_idx[column]
    
    # Replaces beginning zeros and ending zeros with nan if any
    if first_non_zero is not None and last_non_zero is not None:
        df_cleaning.loc[:first_non_zero-1, column] = np.nan
        df_cleaning.loc[last_non_zero+1:, column] = np.nan

# Print the removed datapoints
print(f'Original datapoints: {df_original.count().sum()}')
print(f'Cleaned datapoints: {df_cleaning.count().sum()}')
print(f'Datapoints removed: {df_original.count().sum() - df_cleaning.count().sum()}')

# Readds the date and time before transfering to csv file
df_cleaning = pd.concat([df[['Dato', 'Time']], df_cleaning], axis=1)

# Return the cleaned data into another csv
df_cleaning.to_csv('trefor_cleaned.csv', index=False)