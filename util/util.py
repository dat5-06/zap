from pathlib import Path
import pandas as pd


def read_csv(file_path: str) -> pd.DataFrame:
    """Read csv file to pandas dataframe.

    Arguments:
        **kwargs: hey
        file_path: Name of csv file to be read.

    """
    path = Path(__file__).parent / file_path
    return pd.read_csv(path, sep=";", decimal=",")


def write_csv(df: pd.DataFrame, file_path: str) -> None:
    """Write csv file to specified relative path.

    Arguments:
        file_path: relative output path for the csv file.
        df: DataFrame to convert to csv.

    """
    path = Path(__file__).parent / file_path
    df.to_csv(path, sep=";", decimal=",", index=False)
