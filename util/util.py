from pathlib import Path
import pandas as pd


def read_csv(file_path: str) -> pd.DataFrame:
    """Read csv file to pandas dataframe.

    Arguments:
        file_path: Path to csv relative to project root file to be read.

    """
    project_root = get_project_root()
    path = project_root / file_path
    return pd.read_csv(path, sep=";", decimal=",")


def write_csv(df: pd.DataFrame, file_path: str) -> None:
    """Write csv file to specified path.

    Arguments:
        file_path: relative output path from project root for the csv file.
        df: DataFrame to convert to csv.

    """
    project_root = get_project_root()
    path = project_root / file_path
    df.to_csv(path, sep=";", decimal=",", index=False)


def get_project_root() -> Path:
    """Return path of the project root directory."""
    cur_path = Path(__file__).resolve()
    root_dir = "ev-charging"

    # Traverse up the directories until root directory is reached
    for parent in cur_path.parents:
        if parent.name == root_dir:
            return parent

    # Raise exception if root directory is not found
    raise FileNotFoundError(
        root_dir + " " + "directory was not found in the path hierarchy."
    )
