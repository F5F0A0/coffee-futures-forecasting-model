"""Data loading and backtest utilities for the coffee forecast project."""

import pandas as pd


def load_coffee_data(file_path="../data/coffee.csv"):
    """Load and clean the ICE Coffee C futures dataset.

    Returns a DataFrame with columns [unique_id, ds, y], sorted by date,
    with dates normalized and non-numeric prices dropped.
    """
    df = pd.read_csv(file_path)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    df["ds"] = pd.to_datetime(df["ds"]).dt.normalize()
    df = df.sort_values("ds").reset_index(drop=True)
    if "unique_id" not in df.columns:
        df["unique_id"] = "coffee"
    return df


def get_anchor_points(df, num_windows, context_len=1536, horizon=63):
    """Evenly spaced forecast-origin indices for the rolling backtest.

    Returns indices where each origin has at least `context_len` days of
    history behind it and at least `horizon` days of data ahead of it.
    """
    N = len(df)
    start = context_len
    max_end = N - horizon
    if num_windows == 1:
        return [max_end]
    step = (max_end - start) // (num_windows - 1)
    return [start + i * step for i in range(num_windows)]
