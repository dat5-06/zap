from typing import Iterator
import torch
import numpy as np

from core.util.io import read_csv


def get_one_park_dataset(
    lookback: int, horizon: int, park_number: int, features: dict, folds: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Get normalized train-, val- and test datasets for Trefor parks."""
    park = read_csv(f"processed/park_{park_number}.csv")
    drop_columns = [
        j
        for j in list(park.columns)
        if features.get(j) is None or features.get(j) is False
    ]
    drop_columns.remove("Consumption")  # Ensure "consumption" column is not dropped
    park = park.drop(drop_columns, axis=1)
    x, y = split_sequences(park.to_numpy(), park.to_numpy(), lookback, horizon)

    x_train, y_train, x_test, y_test, indexes = get_one_cross_park(x, y, folds)

    return (x_train, y_train, x_test, y_test, indexes)


def get_one_cross_park(
    x: np.ndarray, y: np.ndarray, folds_num: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Split the data for cross-validation into one array and split indices."""
    # Create indexes for 80% training and 10% validation
    split_test = int(len(x) * 0.9)
    x_train_val, y_train_val, x_test, y_test = (
        x[:split_test],
        y[:split_test],
        x[split_test:],
        y[split_test:],
    )

    # Define fold sizes
    fold_size = len(x_train_val) // folds_num
    fold_indices = [
        (i * fold_size, min((i + 1) * fold_size, len(x_train_val)))
        for i in range(folds_num)
    ]

    return x_train_val, y_train_val, x_test, y_test, fold_indices


def split_sequences(
    features: np.ndarray, targets: np.ndarray, lookback: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split a multivaritae sequence past, future samples."""
    x = []
    y = []
    for i in range(len(features)):
        # Get the lookback / forward window
        lookback_index = i + lookback
        fwd_index = lookback_index + horizon
        # check if we are out of bounds
        if fwd_index > len(features):
            break
        seq_x, seq_y = features[i:lookback_index], targets[lookback_index:fwd_index, -1]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def cross_validation(
    lookback: int, horizon: int, folds: int, features: dict = {}
) -> Iterator[
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        np.ndarray,
    ]
]:
    """Generate permutations for cross-validation."""
    x_trains, y_trains = [], []
    x_tests, y_tests = [], []
    all_fold_indices = []

    # Iterate over parks
    for i in range(1, 7):  # Assuming 5 parks
        x_train_val, y_train_val, x_test, y_test, fold_indices = get_one_park_dataset(
            lookback, horizon, i, features, folds=folds
        )
        x_trains.append(x_train_val)
        y_trains.append(y_train_val)
        x_tests.append(x_test)
        y_tests.append(y_test)
        all_fold_indices.append(fold_indices)

    x_test = torch.tensor(np.concatenate(x_tests)).float()
    y_test = torch.tensor(np.concatenate(y_tests)).float()

    # For each fold
    for k in range(folds):
        x_train, y_train, x_val, y_val = [], [], [], []

        for i in range(6):  # Assuming 5 parks
            start, end = all_fold_indices[i][k]
            split = int(start + ((end - start) * 0.9))
            print(start, split, end)
            # Training set includes only data before the validation block
            x_train.append(x_trains[i][start:split])
            y_train.append(y_trains[i][start:split])

            # Validation set is the block defined by the current fold
            x_val.append(x_trains[i][split:end])
            y_val.append(y_trains[i][split:end])

        yield (
            torch.tensor(np.concatenate(x_train)).float(),
            torch.tensor(np.concatenate(y_train)).float(),
            torch.tensor(np.concatenate(x_val)).float(),
            torch.tensor(np.concatenate(y_val)).float(),
            x_test,
            y_test,
            np.array([len(x) for x in x_tests]),  # Test indices
        )
