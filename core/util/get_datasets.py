from typing import Iterator
import torch
import numpy as np

from core.util.io import read_csv


def get_one_park_dataset(park_number: int, features: dict) -> np.ndarray:
    """Get normalized train-, val- and test datasets for Trefor parks."""
    park = read_csv(f"processed/park_{park_number}.csv")
    drop_columns = [
        j
        for j in list(park.columns)
        if (features.get(j) is None or features.get(j) is False)
        and j != "Consumption"  # ensure consumption is not dropped
    ]
    park = park.drop(drop_columns, axis=1)

    return park.to_numpy()


def get_one_cross_park(x: np.ndarray, folds_num: int) -> list[tuple[int, int]]:
    """Split the data for cross-validation into one array and split indices."""
    # Define fold sizes
    fold_size = len(x) // folds_num
    fold_indices = [
        (i * fold_size, min((i + 1) * fold_size, len(x))) for i in range(folds_num)
    ]

    return fold_indices


def split_sequences(
    features: np.ndarray, lookback: int, horizon: int
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
        seq_x, seq_y = (
            features[i:lookback_index],
            features[lookback_index:fwd_index, -1],
        )
        x.append(seq_x)
        y.append(seq_y)
    return (np.array(x), np.array(y))


def cross_validation(
    lookback: int,
    horizon: int,
    train_days: int,
    val_days: int,
    test_days: int,
    features: dict = {},
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
    parks = []

    # Iterate over parks
    for i in range(1, 7):
        park = get_one_park_dataset(i, features)
        parks.append(park)

    x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []

    simon_length = 0
    for p in parks:
        if len(p) > simon_length:
            simon_length = len(p)

    # For each fold
    train_days *= 24
    val_days *= 24
    test_days *= 24
    diff = train_days + val_days + test_days
    length = simon_length // diff

    for i in range(length - 1):
        for j in range(6):
            train_start = i * length
            train_end = train_start + train_days
            val_start = train_end
            val_end = val_start + val_days
            test_start = val_end
            test_end = test_start + test_days

            if len(parks[j]) < test_end:
                continue

            # Training set includes only data before the validation block
            x_train_split, y_train_split = split_sequences(
                parks[j][train_start:train_end],
                lookback=lookback,
                horizon=horizon,
            )
            x_train.append(x_train_split)
            y_train.append(y_train_split)

            # Validation set is the block defined by the current fold
            x_val_split, y_val_split = split_sequences(
                parks[j][val_start:val_end],
                lookback=lookback,
                horizon=horizon,
            )
            x_val.append(x_val_split)
            y_val.append(y_val_split)

            x_test_split, y_test_split = split_sequences(
                parks[j][test_start:test_end],
                lookback=lookback,
                horizon=horizon,
            )
            x_test.append(x_test_split)
            y_test.append(y_test_split)

    return (
        torch.tensor(np.concatenate(x_train)).float(),
        torch.tensor(np.concatenate(y_train)).float(),
        torch.tensor(np.concatenate(x_val)).float(),
        torch.tensor(np.concatenate(y_val)).float(),
        torch.tensor(np.concatenate(x_test)).float(),
        torch.tensor(np.concatenate(y_test)).float(),
        np.array([len(x) for x in x_test]),  # Test indices
    )
