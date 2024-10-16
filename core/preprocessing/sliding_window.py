import numpy as np
from numpy._typing import _UnknownType
import pandas as pd


def apply_sliding(
    timeseries: pd.DataFrame | pd.Series | _UnknownType, backward: int, forward: int
) -> tuple[tuple[np.ndarray, np.ndarray], int]:
    """Apply sliding window of size n.

    Arguments:
    ---------
        timeseries: The time series data
        backward: Size of the sliding window
        forward: Number of steps to predict into the future

    """
    valid_index = int(timeseries.notna().idxmax())

    # lookback window
    x = np.array(
        [
            timeseries[i - backward : i]
            for i in range(backward + valid_index, len(timeseries) - forward + 1)
        ]
    )

    # targets
    y = np.array(
        [
            timeseries[i : i + forward]
            for i in range(backward + valid_index, len(timeseries) - forward + 1)
        ]
    )
    return (x, y), valid_index
