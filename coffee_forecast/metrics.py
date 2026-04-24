"""
Point-forecast accuracy metrics.

All five metrics are computed for every (model, origin) pair in the
backtest. Keeping them in one function guarantees identical formulas
across every table and figure in the paper.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    train_y: np.ndarray,
) -> Dict[str, float]:
    """
    Compute a suite of point-forecast accuracy metrics.

    Parameters
    ----------
    y_true : array-like
        Realized values over the forecast horizon.
    y_hat : array-like
        Point forecasts over the same horizon (same length as ``y_true``).
    train_y : array-like
        The context / training window, used only to compute the naive-forecast
        scale for MASE. Must have length >= 2.

    Returns
    -------
    dict
        Keys: ``MAE``, ``RMSE``, ``MAPE``, ``sMAPE``, ``MASE``.

    Notes
    -----
    MASE here uses the in-sample naive-forecast scale
    ``mean(|y_t - y_{t-1}|)`` on the training window. A MASE of 1.0 means
    the model matches a one-step naive forecast; anything >1 is losing to
    the naive baseline in scale-free terms.
    """
    y_true  = np.asarray(y_true,  dtype=float)
    y_hat   = np.asarray(y_hat,   dtype=float)
    train_y = np.asarray(train_y, dtype=float)

    mae   = mean_absolute_error(y_true, y_hat)
    rmse  = np.sqrt(mean_squared_error(y_true, y_hat))
    mape  = mean_absolute_percentage_error(y_true, y_hat)
    smape = np.mean(
        2.0 * np.abs(y_true - y_hat)
        / (np.abs(y_true) + np.abs(y_hat) + 1e-8)
    )

    # MASE scale: mean absolute first-difference on the training window.
    scale = np.mean(np.abs(np.diff(train_y)))
    mase  = mae / (scale if scale > 0 else 1e-8)

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "MASE": mase}
