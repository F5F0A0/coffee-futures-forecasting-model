"""
Point-forecast accuracy metrics.

All five metrics are computed for every (model, origin) pair in the
backtest. Keeping them in one function guarantees identical formulas
everywhere they're reported.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


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
    MASE uses Hyndman's in-sample naive-forecast scale
    ``mean(|y_t - y_{t-1}|)`` on the training window. Lower is better, and
    values are comparable across models forecasting the same series at the
    same horizon.

    The ``MASE = 1.0`` interpretation as "matches a naive forecast" only
    holds for one-step forecasts on a stationary series. For multi-step
    forecasts on volatile holdouts (here ``HORIZON = 63`` over a series
    that includes the 2025 surge), MASE >> 1 is expected even for the
    naive baseline: errors compound across the horizon and test-window
    volatility may exceed the in-sample volatility used to set the scale.
    Cross-model comparison is unaffected.
    """
    y_true  = np.asarray(y_true,  dtype=float)
    y_hat   = np.asarray(y_hat,   dtype=float)
    train_y = np.asarray(train_y, dtype=float)

    mae   = np.mean(np.abs(y_true - y_hat))
    rmse  = np.sqrt(np.mean((y_true - y_hat) ** 2))
    mape  = np.mean(np.abs((y_true - y_hat) / y_true))
    smape = np.mean(
        2.0 * np.abs(y_true - y_hat)
        / (np.abs(y_true) + np.abs(y_hat) + 1e-8)
    )

    # MASE scale: mean absolute first-difference on the training window.
    scale = np.mean(np.abs(np.diff(train_y)))
    mase  = mae / (scale if scale > 0 else 1e-8)

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "MASE": mase}
