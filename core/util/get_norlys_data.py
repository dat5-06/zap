from typing import Iterator
import torch
import numpy as np

from core.util.io import read_csv


def get_one_park_dataset(consumer: bool, features: dict) -> np.ndarray:
    """Get normalized train-, val- and test datasets for Trefor parks."""
    data_type = "pub"
    if consumer:
        data_type = "cons"

    park = read_csv(f"processed/norlys/norlys_{data_type}.csv")
    drop_columns = [
        j
        for j in list(park.columns)
        if (features.get(j) is None or features.get(j) is False)
        and j != "kwh_percentage"  # ensure consumption is not dropped
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
    consumer: bool,
    features: dict = {},
) -> Iterator[
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
]:
    """Generate permutations for cross-validation."""
    # Iterate over parks
    park = get_one_park_dataset(consumer, features)

    x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []

    max_length = len(park)

    # For each fold
    train_days *= 24
    val_days *= 24
    test_days *= 24
    diff = train_days + val_days + test_days + lookback + (24 * 3)
    length = max_length // diff

    for i in range(length - 1):
        train_start = i * length
        train_end = train_start + train_days + lookback + 24
        val_start = train_end - lookback
        val_end = train_end + val_days + 24
        test_start = val_end - lookback
        test_end = val_end + test_days + 24

        if len(park) < test_end:
            continue

        # Training set includes only data before the validation block
        x_train_split, y_train_split = split_sequences(
            park[train_start:train_end],
            lookback=lookback,
            horizon=horizon,
        )

        x_train.append(x_train_split)
        y_train.append(y_train_split)

        # Validation set is the block defined by the current fold
        x_val_split, y_val_split = split_sequences(
            park[val_start:val_end],
            lookback=lookback,
            horizon=horizon,
        )

        x_val.append(x_val_split)
        y_val.append(y_val_split)

        x_test_split, y_test_split = split_sequences(
            park[test_start:test_end],
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
    )
