"""
Data I/O for the ICE Coffee C futures series.

Kept deliberately small: anything that transforms the series beyond basic
cleaning (log returns, log-diffs, GARCH inputs, ...) lives in the caller,
not here, so the raw-series contract is easy to audit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from .config import DEFAULT_DATA_FILE


def load_coffee_data(file_path: Union[str, Path, None] = None) -> pd.DataFrame:
    """
    Load the ICE Coffee C daily close series.

    Parameters
    ----------
    file_path : str | Path | None
        Path to the CSV. Defaults to ``data/coffee.csv`` relative to the
        package root, so the loader works no matter where you run it from.

    Returns
    -------
    pd.DataFrame
        Columns: ``ds`` (datetime, normalized to midnight), ``y`` (float,
        price in cents/lb), ``unique_id`` (string, always ``"coffee"`` here
        - required by statsforecast's API).

    Notes
    -----
    - Cleaning (numeric coercion, dropping the XLS header rows) happens
      once in ``scripts/export_csv.py``. This loader trusts the resulting
      CSV and raises on malformed rows rather than silently masking them,
      so a corrupted CSV surfaces immediately.
    - Dates are normalized so downstream date arithmetic never trips over
      mixed 00:00:00 / business-hours timestamps.
    """
    path = Path(file_path) if file_path is not None else DEFAULT_DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Coffee data file not found at {path}. "
            "Expected data/coffee.csv at the repo root."
        )

    df = pd.read_csv(path)

    # Trust the cleaned CSV; raise (don't coerce) if a row is malformed.
    df["y"]  = pd.to_numeric(df["y"])
    df["ds"] = pd.to_datetime(df["ds"]).dt.normalize()

    # Chronological ordering is assumed by every downstream function.
    df = df.sort_values("ds").reset_index(drop=True)

    # statsforecast requires a unique_id column; hard-coded for a single series.
    if "unique_id" not in df.columns:
        df["unique_id"] = "coffee"

    return df
