"""
Thin wrappers that give every forecasting library the same interface:
``predict(context_df, horizon) -> np.ndarray`` of shape ``(horizon,)``.

The uniform API is what lets the backtest runner treat Granite, ARIMA,
GARCH, Prophet, Ridge, and Random Forest as interchangeable black boxes.
"""
from __future__ import annotations

import warnings
from typing import List, Sequence, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class ModelWrapper:
    """Abstract base class. All wrappers implement ``predict``."""

    def predict(self, context_df: pd.DataFrame, horizon: int) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# IBM Granite Tiny Time Mixer (foundation model, zero-shot)
# ---------------------------------------------------------------------------
class GraniteWrapper(ModelWrapper):
    """
    Zero-shot forecasting with the IBM Granite TTM foundation model.

    The model expects a fixed-length context window and returns a fixed
    prediction length; we standardize the input per-window (z-score against
    the context mean/std) and undo the scaling on the output.
    """

    def __init__(self, model, device: str, context_len: int = 1536) -> None:
        self.model       = model
        self.device      = device
        self.context_len = context_len

    def predict(self, context_df: pd.DataFrame, horizon: int) -> np.ndarray:
        import torch  # local import: torch is heavy and only needed here

        ctx = context_df["y"].values[-self.context_len:].astype(np.float32)

        # Standard-scale the context so the foundation model sees the
        # distribution it was pretrained on.
        mu    = ctx.mean()
        sigma = ctx.std()
        scaled = (ctx - mu) / (sigma if sigma > 0 else 1.0)

        past_t = torch.tensor(scaled, device=self.device).view(1, -1, 1)
        with torch.no_grad():
            out = self.model(past_values=past_t)
        pred = out.prediction_outputs.detach().cpu().numpy()[0, :, 0]

        # Undo the per-window standardization.
        return (pred * sigma + mu)[:horizon]


# ---------------------------------------------------------------------------
# statsforecast models (Naive, RWD, Theta, AutoETS, AutoARIMA)
# ---------------------------------------------------------------------------
class StatsForecastWrapper(ModelWrapper):
    """
    Wraps a single ``statsforecast`` model. Accepts either the model
    directly or a 1-element list for convenience.
    """

    def __init__(self, models, freq: str = "B") -> None:
        if isinstance(models, list):
            if len(models) != 1:
                raise ValueError(
                    "StatsForecastWrapper currently wraps one model at a time."
                )
            self.models = models
        else:
            self.models = [models]
        self.freq = freq

    def predict(self, context_df: pd.DataFrame, horizon: int) -> np.ndarray:
        from statsforecast import StatsForecast

        sf = StatsForecast(models=self.models, freq=self.freq, n_jobs=1)
        sf.fit(context_df[["unique_id", "ds", "y"]])
        fcst = sf.predict(h=horizon)

        # Every statsforecast model exposes its column either via ``alias``
        # or its class name. Using ``alias`` is the safer default.
        model = self.models[0]
        model_name = model.alias if hasattr(model, "alias") else str(model)
        return fcst[model_name].values


# ---------------------------------------------------------------------------
# Recursive ML wrapper (Ridge, Random Forest)
# ---------------------------------------------------------------------------
class MLRecursiveWrapper(ModelWrapper):
    """
    Recursive multi-step forecaster over lag + rolling-window features.

    Training set is built once per call from the context window, then the
    fitted estimator is called ``horizon`` times, feeding its own prediction
    back in as the next observation.
    """

    def __init__(
        self,
        model,
        lags: Sequence[int] = (1, 2, 5, 10, 21, 63),
        rolls: Sequence[int] = (5, 21, 63),
    ) -> None:
        self.model = model
        self.lags  = list(lags)
        self.rolls = list(rolls)

    def _feature_vector(self, y_series: np.ndarray) -> dict:
        """One row of features from the history up to the last observed point."""
        feat: dict = {}
        for L in self.lags:
            feat[f"lag_{L}"] = y_series[-L] if len(y_series) >= L else 0.0
        for w in self.rolls:
            window = y_series[-w:] if len(y_series) >= w else y_series
            feat[f"roll_mean_{w}"] = float(np.mean(window))
            feat[f"roll_std_{w}"]  = float(np.std(window)) if len(window) > 1 else 0.0
        return feat

    def predict(self, context_df: pd.DataFrame, horizon: int) -> np.ndarray:
        y = context_df["y"].values.astype(float)

        # Build training set: for every t, features built from y[:t+1]
        # predict y[t+1]. Skip the early prefix where lags are undefined.
        min_history = max(max(self.lags), max(self.rolls))
        X_rows, y_rows = [], []
        for i in range(min_history, len(y) - 1):
            X_rows.append(self._feature_vector(y[: i + 1]))
            y_rows.append(y[i + 1])

        X_train = pd.DataFrame(X_rows)
        self.model.fit(X_train, y_rows)

        # Recursive forecast: extend the history with each prediction.
        preds: List[float] = []
        history = list(y)
        feat_cols = X_train.columns  # enforce column order on each step
        for _ in range(horizon):
            X_next = pd.DataFrame([self._feature_vector(np.asarray(history))],
                                  columns=feat_cols)
            y_hat = float(self.model.predict(X_next)[0])
            preds.append(y_hat)
            history.append(y_hat)

        return np.array(preds)


# ---------------------------------------------------------------------------
# GARCH (volatility model, used here for its mean-equation forecast)
# ---------------------------------------------------------------------------
class GARCHWrapper(ModelWrapper):
    """
    AR(1) mean + GARCH(1,1) volatility, fit on raw prices.

    We use the mean forecast as the point forecast; the volatility forecast
    is used separately in the deployment notebook to build prediction
    intervals, not in this backtest.
    """

    def predict(self, context_df: pd.DataFrame, horizon: int) -> np.ndarray:
        from arch import arch_model
        from arch.univariate.base import DataScaleWarning

        y = context_df["y"].values.astype(float)
        am = arch_model(y, vol="Garch", p=1, q=1, mean="AR", lags=1, dist="Normal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataScaleWarning)
            res = am.fit(disp="off")
        return res.forecast(horizon=horizon).mean.iloc[-1].values


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------
class ProphetWrapper(ModelWrapper):
    """Facebook Prophet with yearly seasonality, no daily seasonality."""

    def predict(self, context_df: pd.DataFrame, horizon: int) -> np.ndarray:
        from prophet import Prophet

        m = Prophet(daily_seasonality=False, yearly_seasonality=True)
        m.fit(context_df[["ds", "y"]])
        future = m.make_future_dataframe(periods=horizon, freq="B")
        return m.predict(future)["yhat"].iloc[-horizon:].values
