"""Preprocess all data."""

from core.preprocessing.acn import acn
from core.preprocessing.trefor import trefor


def preprocess() -> None:
    """Preprocess all data."""
    acn()
    trefor()


if __name__ == "__main__":
    preprocess()
