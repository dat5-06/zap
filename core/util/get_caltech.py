import numpy as np
import torch

from core.util.io import read_csv
from core.util.get_datasets import split_sequences


def get_caltech_dataset(features: dict) -> np.ndarray:
    """Get normalized train-, val- and test datasets for Trefor parks."""
    park = read_csv("processed/caltech_ev_sessions.csv")
    drop_columns = [
        col
        for col in list(park.columns)
        if (features.get(col) is None or features.get(col) is False)
        and col != "Consumption"  # ensure consumption is not dropped
    ]
    park = park.drop(drop_columns, axis=1)

    return park.to_numpy()


def caltech_cross_validation(
    lookback: int,
    horizon: int,
    train_days: int,
    val_days: int,
    test_days: int,
    block_gap: int = 0,
    features: dict = {},
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Divide dataset into training, validation and test sets using some block size."""
    park = get_caltech_dataset(features)

    x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []

    # We change from days to hours
    train_size = train_days * 24
    val_size = val_days * 24
    test_size = test_days * 24

    # For each block we add padding for the training, validation and test
    # This is equal to the lookback + horizon + block_gap
    train_length = train_size + lookback + horizon + block_gap
    val_length = val_size + lookback + horizon + block_gap
    test_length = test_size + lookback + horizon + block_gap

    block_length = train_length + val_length + test_length

    # calculate the amount of blocks that fit into the dataset
    num_blocks = len(park) // block_length

    # iterate the blocks
    for block_num in range(num_blocks):
        # The indicies are the same for all parks
        train_start = block_num * block_length
        train_end = train_start + train_size + lookback + horizon

        val_start = train_end + block_gap
        val_end = train_end + val_size + lookback + horizon

        test_start = val_end + block_gap
        test_end = val_end + test_size + lookback + horizon

        # now we can iterate the parks

        # Now, we can apply sliding window on the data and use it
        x_train_split, y_train_split = split_sequences(
            park[train_start:train_end],
            lookback=lookback,
            horizon=horizon,
        )
        x_train.append(x_train_split)
        y_train.append(y_train_split)

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

    # Join the sequences of blocks into one big blob of data (block)
    return (
        torch.tensor(np.concatenate(x_train)).float(),
        torch.tensor(np.concatenate(y_train)).float(),
        torch.tensor(np.concatenate(x_val)).float(),
        torch.tensor(np.concatenate(y_val)).float(),
        torch.tensor(np.concatenate(x_test)).float(),
        torch.tensor(np.concatenate(y_test)).float(),
    )
