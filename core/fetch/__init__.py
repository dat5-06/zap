"""Fetching sources from the web."""

from core.fetch.dmi import fetch_dmi
from core.fetch.acn import fetch_acn
from core.fetch.eds import fetch_eds


def fetch() -> None:
    """Fetch all datasources."""
    fetch_dmi()
    fetch_acn()
    fetch_eds()
