"""
Rolling-window backtest mechanics.

One-stop shop for:
  - ``get_forecast_origins``      : picks N evenly-spaced forecast origins
  - ``run_test``                  : backtests a single model
  - ``run_multi_scale_backtest``  : loops run_test across scales x models

The multi-scale runner is what the original ``Main.txt`` did inline; pulling
it out turns five indented levels of loops into one function call.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import CONTEXT_LEN, HORIZON
from .metrics import calculate_metrics
from .models import ModelWrapper


# ---------------------------------------------------------------------------
# Forecast-origin placement
# ---------------------------------------------------------------------------
def get_forecast_origins(
    df: pd.DataFrame,
    num_windows: int,
    context_len: int = CONTEXT_LEN,
    horizon: int = HORIZON,
) -> List[int]:
    """
    Choose evenly-spaced forecast-origin indices inside the series.

    The first candidate origin sits at ``context_len`` (you need that much
    history to fit any model); the last sits at ``len(df) - horizon``
    (you need that much future to score it). Everything in between is
    evenly spaced. ``num_windows == 1`` returns the *latest* valid origin,
    which is what you want for a single "final holdout" test.
    """
    N       = len(df)
    start   = context_len
    max_end = N - horizon

    if num_windows < 1:
        raise ValueError("num_windows must be >= 1")
    if max_end <= start:
        raise ValueError(
            f"Series of length {N} is too short for context_len={context_len} "
            f"and horizon={horizon}."
        )

    if num_windows == 1:
        return [max_end]

    step = (max_end - start) // (num_windows - 1)
    return [start + i * step for i in range(num_windows)]


# ---------------------------------------------------------------------------
# Single-model backtest
# ---------------------------------------------------------------------------
def run_test(
    df: pd.DataFrame,
    model_wrapper: ModelWrapper,
    num_windows: int,
    context_len: int = CONTEXT_LEN,
    horizon: int = HORIZON,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Backtest one model across ``num_windows`` forecast origins.

    Parameters
    ----------
    df : DataFrame
        Full series with ``ds``, ``y`` columns (as produced by
        ``load_coffee_data``).
    model_wrapper : ModelWrapper
        Any object exposing ``predict(context_df, horizon) -> array``.
    num_windows : int
        Number of forecast origins (e.g. 1, 10, 30, 60).
    context_len, horizon : int
        Training window length and forecast horizon, in trading days.

    Returns
    -------
    summary_df : DataFrame
        One row per origin, columns:
        ``MAE, RMSE, MAPE, sMAPE, MASE, origin_id, origin_date``.
    per_step_errors : ndarray, shape (num_windows, horizon)
        Absolute errors at each forecast step, for every origin.
        This is what the Diebold-Mariano / MCS tests consume downstream.
    """
    origins = get_forecast_origins(df, num_windows, context_len, horizon)
    summary_rows: List[dict] = []
    per_step_errors: List[np.ndarray] = []

    for i, a in enumerate(origins):
        train_window = df.iloc[a - context_len : a]
        test_window  = df.iloc[a : a + horizon]

        y_hat  = np.asarray(model_wrapper.predict(train_window, horizon), dtype=float)
        y_true = test_window["y"].values.astype(float)

        if len(y_hat) != horizon:
            raise ValueError(
                f"Model returned {len(y_hat)} predictions but horizon={horizon}. "
                f"Check the wrapper for origin {i + 1}."
            )

        metrics = calculate_metrics(y_true, y_hat, train_window["y"].values)
        metrics.update({
            "origin_id":   i + 1,
            "origin_date": df.iloc[a]["ds"],
        })
        summary_rows.append(metrics)
        per_step_errors.append(np.abs(y_hat - y_true))

    return pd.DataFrame(summary_rows), np.asarray(per_step_errors)


# ---------------------------------------------------------------------------
# Multi-scale, multi-model orchestrator
# ---------------------------------------------------------------------------
def run_multi_scale_backtest(
    df: pd.DataFrame,
    models: Dict[str, ModelWrapper],
    scales: List[int],
    context_len: int = CONTEXT_LEN,
    horizon: int = HORIZON,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, np.ndarray]]]:
    """
    Run every model at every scale, collect everything into a tidy long frame.

    This is the workhorse that used to live inline at the top of
    ``Main.txt``. Result shape is unchanged so the downstream leaderboard,
    DM, and MCS code all still work verbatim.

    Parameters
    ----------
    df : DataFrame
        Full series from ``load_coffee_data``.
    models : dict[str, ModelWrapper]
        Model name -> wrapper. Name is used as the ``model`` column in
        the summary and as the key in the step-errors dict.
    scales : list[int]
        Number of forecast origins to evaluate at each scale.
    context_len, horizon : int
        Passed through to ``run_test``.
    verbose : bool
        If True, prints progress per (scale, model).

    Returns
    -------
    summary : DataFrame
        Long format. Columns:
        ``MAE, RMSE, MAPE, sMAPE, MASE, origin_id, origin_date, model, scale``.
    step_errors : dict[int, dict[str, ndarray]]
        ``step_errors[scale][model_name]`` -> array of shape
        ``(num_origins, horizon)`` with per-step absolute errors.
    """
    all_summaries: List[pd.DataFrame] = []
    all_step_errors: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in scales}

    for scale in scales:
        if verbose:
            print(f"\nRunning {scale} forecast origin(s)...")
        for name, wrapper in models.items():
            try:
                res_df, step_errors = run_test(
                    df, wrapper,
                    num_windows=scale,
                    context_len=context_len,
                    horizon=horizon,
                )
                res_df["model"] = name
                res_df["scale"] = scale
                all_summaries.append(res_df)
                all_step_errors[scale][name] = step_errors
                if verbose:
                    print(f"  \u2713 {name}")
            except Exception as e:                      # keep one failure local
                if verbose:
                    print(f"  \u2717 {name}: {e}")

    summary = pd.concat(all_summaries, ignore_index=True)
    return summary, all_step_errors


# ---------------------------------------------------------------------------
# Serialization helper for per-step errors
# ---------------------------------------------------------------------------
def step_errors_to_long_df(
    step_errors: Dict[int, Dict[str, np.ndarray]],
) -> pd.DataFrame:
    """
    Flatten the nested ``{scale: {model: ndarray}}`` structure into a tidy
    long DataFrame. One row per (scale, model, origin, step).

    This is the format the original ``step_errors_all_scales.csv`` uses,
    and what ``diebold_mariano_test`` / ``model_confidence_set`` expect
    downstream.
    """
    rows: List[dict] = []
    for scale, model_errors in step_errors.items():
        for model_name, errors in model_errors.items():
            for origin_idx, origin_errors in enumerate(errors):
                for step_idx, err in enumerate(origin_errors):
                    rows.append({
                        "scale":     scale,
                        "model":     model_name,
                        "origin_id": origin_idx + 1,
                        "step":      step_idx + 1,
                        "error":     float(err),
                    })
    return pd.DataFrame(rows)
