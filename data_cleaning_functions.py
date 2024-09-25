# Finds all data greater than 0 and returns the first index of where a datapoint that isn't zero occurs
def find_first_non_zero(series):
    non_zero_indices = series[series > 0].index
    if len(non_zero_indices) > 0:
        return non_zero_indices[0] 
    else:
        return None  

# Finds all data greater than 0 and returns the last index of where a datapoint that isn't zero occurs
def find_last_non_zero(series):
    non_zero_indices = series[series > 0].index
    if len(non_zero_indices) > 0:
        return non_zero_indices[-1]  
    else:
        return None
