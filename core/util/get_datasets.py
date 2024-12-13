import torch
import numpy as np

from core.util.io import read_csv
from core.preprocessing.trefor import capacities


def get_one_park_dataset(park_number: int, features: dict) -> np.ndarray:
    """Get normalized train-, val- and test datasets for Trefor parks."""
    park = read_csv(f"processed/park_{park_number}.csv")
    drop_columns = [
        col
        for col in list(park.columns)
        if (features.get(col) is None or features.get(col) is False)
        and col != "Consumption"  # ensure consumption is not dropped
    ]
    park = park.drop(drop_columns, axis=1)

    return park.to_numpy()


def split_sequences(
    sequence: np.ndarray, lookback: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply sliding window on a sequence of data, given a lookback and horizon."""
    x = []
    y = []
    # iterate the sequence to give `len(sequence) - lookback - horizon` sized output
    for i in range(len(sequence) - lookback - horizon):
        # get the lookback index
        lookback_index = i + lookback
        # get horizon index
        fwd_index = lookback_index + horizon

        # x is all the features, and y is only the consumption (last column)
        seq_x, seq_y = (
            sequence[i:lookback_index],
            sequence[lookback_index:fwd_index, -1],
        )
        x.append(seq_x)
        y.append(seq_y)
    return (np.array(x), np.array(y))


def split_data(
    lookback: int,
    horizon: int,
    train_days: int,
    val_days: int,
    test_days: int,
    features: dict = {},
    park_nums: list = [*range(1, 7)],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Divide dataset into training, validation and test sets using some block size."""
    parks = [get_one_park_dataset(i, features) for i in park_nums]  # park 1 through 6

    x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []

    # We change from days to hours
    train_size = train_days * 24
    val_size = val_days * 24
    test_size = test_days * 24

    # For each block we add padding for the training, validation and test
    # This is equal to the lookback + horizon
    lookback_shift = 96 - lookback
    train_length = train_size + lookback + horizon + lookback_shift
    val_length = val_size + lookback + horizon + lookback_shift
    test_length = test_size + lookback + horizon + lookback_shift
    block_length = train_length + val_length + test_length

    # calculate the amount of blocks that fit into the dataset of a single park
    biggest_park_size = max(len(park) for park in parks)
    num_blocks = biggest_park_size // block_length

    # iterate the blocks
    for block_num in range(num_blocks):
        # The indicies are the same for all parks
        train_start = block_num * block_length + lookback_shift
        train_end = train_start + train_size + lookback + horizon

        val_start = train_end + lookback_shift
        val_end = val_start + val_size + lookback + horizon

        test_start = val_end + lookback_shift
        test_end = test_start + test_size + lookback + horizon

        # now we can iterate the parks
        for park_index in range(len(park_nums)):
            # the parks are not equal in length, so maybe there is no more data left
            # if that is the case, we skip it
            if len(parks[park_index]) < test_end:
                continue

            # Now, we can apply sliding window on the data and use it
            x_train_split, y_train_split = split_sequences(
                parks[park_index][train_start:train_end],
                lookback=lookback,
                horizon=horizon,
            )
            x_train.append(x_train_split)
            y_train.append(y_train_split)

            x_val_split, y_val_split = split_sequences(
                parks[park_index][val_start:val_end],
                lookback=lookback,
                horizon=horizon,
            )
            x_val.append(x_val_split)
            y_val.append(y_val_split)

            x_test_split, y_test_split = split_sequences(
                parks[park_index][test_start:test_end],
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


def denormalize_data(
    y_test: torch.Tensor | np.ndarray | list,
    train_days: int = 16,
    val_days: int = 2,
    test_days: int = 2,
    horizon: int = 24,
    park_nums: list = [*range(1, 7)],
) -> torch.Tensor:
    """Denormalize the input dataset."""
    assert len(y_test) % (24 * test_days) == 0

    y_test = np.array(y_test)

    parks = [get_one_park_dataset(i, {}) for i in park_nums]  # park 1 through 6

    # we do not need the intermediate variables from the function above
    block_length = (train_days + val_days + test_days) * 24 + (96 + horizon) * 3

    biggest_park_size = max(len(park) for park in parks)
    num_blocks = biggest_park_size // block_length

    park_belongings = []
    for block_num in range(1, num_blocks + 1):
        end_index = block_num * block_length
        # iterate parks
        for i, park in enumerate(parks):
            # determine if a block fits
            if len(park) >= end_index:
                park_belongings.append(i)

    hours = 24 * test_days
    for i, capacity in enumerate(park_belongings):
        y_test[i * hours : (i + 1) * hours, :] = (
            y_test[i * hours : (i + 1) * hours, :] * capacities[capacity]
        )

    return torch.tensor(y_test).float()
