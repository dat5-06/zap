import torch
import numpy as np

from core.util.io import read_csv


def get_one_park_dataset(
    lookback: int, lookahead: int, park_number: int, features: dict, folds: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get normalized train-, val- and test datasets for Trefor parks."""
    park = read_csv(f"processed/park_{park_number}.csv")
    drop_columns = [
        j
        for j in list(park.columns)
        if features.get(j) is None or features.get(j) is False
    ]
    drop_columns.remove("Consumption")  # Ensure "consumption" column is not dropped
    park = park.drop(drop_columns, axis=1)
    x, y = split_sequences(park.to_numpy(), park.to_numpy(), lookback, lookahead)

    # Initialise empty arrays
    x_train, x_val, x_test = [], [], []
    y_train, y_val, y_test = [], [], []

    # Check if the data should be cross validation
    if folds is not None:
        x_train, y_train, x_val, y_val, x_test, y_test = get_one_cross_park(x, y, folds)
    else:
        x_train, y_train, x_val, y_val, x_test, y_test = get_one_norm_park(x, y)

    return (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    )


def get_one_cross_park(
    x: np.ndarray, y: np.ndarray, folds_num: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the data for cross-validation."""
    # Create indexes for 80% training, 10% validation, and 10% testing
    split_test = int(len(x) * 0.9)

    # Split into temporary train and test sets
    x_train_temp, y_train_temp, x_test, y_test = (
        x[:split_test],
        y[:split_test],
        x[split_test:],
        y[split_test:],
    )

    # Define folds for cross-validation
    fold_size = len(x_train_temp) // folds_num

    folds = [
        (
            x_train_temp[i * fold_size : (i + 1) * fold_size],
            y_train_temp[i * fold_size : (i + 1) * fold_size],
        )
        for i in range(folds_num)
    ]

    x_train, y_train, x_val, y_val = [], [], [], []

    # Loop over folds and append the training and validation sets in each split
    for i in range(folds_num):
        x_val_fold, y_val_fold = folds[i]
        x_val.append(x_val_fold)
        y_val.append(y_val_fold)

        x_train_fold = [folds[j][0] for j in range(folds_num) if j != i]
        y_train_fold = [folds[j][1] for j in range(folds_num) if j != i]

        x_train.append(np.concatenate(x_train_fold, axis=0))
        y_train.append(np.concatenate(y_train_fold, axis=0))

    return (
        np.concatenate(x_train, axis=0),
        np.concatenate(y_train, axis=0),
        np.concatenate(x_val, axis=0),
        np.concatenate(y_val, axis=0),
        x_test,
        y_test,
    )


def get_one_norm_park(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the data split of training, validation, and test in that sequence."""
    # Create indexes for 80% training, 10% validation, and 10% testing
    split_test = int(len(x) * 0.9)
    split_val = int(len(x) * 0.8)

    # split into train and test
    x_train, y_train, x_test, y_test = (
        x[:split_test],
        y[:split_test],
        x[split_test:],
        y[split_test:],
    )

    # split train into train and val
    x_train, y_train, x_val, y_val = (
        x_train[:split_val],
        y_train[:split_val],
        x_train[split_val:],
        y_train[split_val:],
    )

    return (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    )


def get_park_datasets(
    lookback: int, lookahead: int, features: dict, folds: int | None
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
]:
    """Get concatenated normalized train-, val- and test datasets for Trefor parks."""
    x_train, y_train = [], []
    x_val, y_val = [], []
    x_test, y_test = [], []
    indicies = []
    combined_len = 0

    # Loop over parks and create their data sets
    for i in range(1, 7):
        x_train_p, y_train_p, x_val_p, y_val_p, x_test_p, y_test_p = (
            get_one_park_dataset(lookback, lookahead, i, features, folds)
        )
        x_train.extend(x_train_p)
        y_train.extend(y_train_p)
        x_val.extend(x_val_p)
        y_val.extend(y_val_p)
        x_test.extend(x_test_p)
        y_test.extend(y_test_p)

        # Get indicies at which splits occur
        split_1 = combined_len + len(x_train_p)
        split_2 = split_1 + len(x_val_p)
        indicies.append([split_1, split_2])

        combined_len += len(x_train_p) + len(x_val_p) + len(x_test_p)

    return (
        torch.Tensor(np.array(x_train)).float(),
        torch.Tensor(np.array(y_train)).float(),
        torch.Tensor(np.array(x_val)).float(),
        torch.Tensor(np.array(y_val)).float(),
        torch.Tensor(np.array(x_test)).float(),
        torch.Tensor(np.array(y_test)).float(),
        np.array(indicies),
    )


def split_sequences(
    features: np.ndarray, targets: np.ndarray, lookback: int, lookahead: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split a multivaritae sequence past, future samples."""
    x = []
    y = []
    for i in range(len(features)):
        # Get the lookback / forward window
        lookback_index = i + lookback
        fwd_index = lookback_index + lookahead
        # check if we are out of bounds
        if fwd_index > len(features):
            break
        seq_x, seq_y = features[i:lookback_index], targets[lookback_index:fwd_index, -1]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def apply_sliding_window(
    timeseries: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply sliding window of size n.

    Arguments:
    ---------
        timeseries: The time series data
        n: Size of the sliding window

    """
    # Initialize lists to hold features and target
    x = []
    y = []

    # Add n datapoints to x then add target to y.
    for i in range(n, len(timeseries)):
        x.append(timeseries[i - n : i])
        y.append([timeseries[i]])
    return np.array(x), np.array(y)


def get_trefor_timeseries() -> np.ndarray:
    """Get processed trefor timeseries data as numpy array."""
    # Read trefor data csv
    data = read_csv("processed/trefor_final.csv")
    # Get just the total consumption, should be further processed with sliding window
    timeseries = data["Total_Consumption"].astype(float).to_numpy().reshape(-1, 1)

    return timeseries


def get_timeseries_dataset(
    timeseries: np.ndarray, n: int
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Get timeseries dataset with 80:10:10 train:val:test split.

    Arguments:
    ---------
        timeseries: The time series data
        n: Size of the sliding window

    """
    # Create sliding window of n size on the timeseries
    x, y = apply_sliding_window(timeseries, n)
    # Create indexes for 80% training, 10% validation, and 10% testing
    split_test = int(len(x) * 0.9)
    split_val = int(len(x) * 0.8)

    # split into train and test
    np_x_train, np_y_train, np_x_test, np_y_test = (
        x[:split_test],
        y[:split_test],
        x[split_test:],
        y[split_test:],
    )

    # split train into train and val
    np_x_train, np_y_train, np_x_val, np_y_val = (
        np_x_train[:split_val],
        np_y_train[:split_val],
        np_x_train[split_val:],
        np_y_train[split_val:],
    )

    # Create tensors from splits
    x_train = torch.tensor(data=np_x_train).float()
    y_train = torch.tensor(data=np_y_train).float()

    x_val = torch.tensor(data=np_x_val).float()
    y_val = torch.tensor(data=np_y_val).float()

    x_test = torch.tensor(data=np_x_test).float()
    y_test = torch.tensor(data=np_y_test).float()

    # Return tensors and squeeze y tensors to fit dimensions
    return (
        x_train,
        x_val,
        x_test,
        y_train.squeeze(1),
        y_val.squeeze(1),
        y_test.squeeze(1),
    )
