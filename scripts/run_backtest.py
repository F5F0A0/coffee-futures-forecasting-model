"""
Reproduces the rolling-window backtest end-to-end and writes:
    results/csv/summary_all_scales.csv
    results/csv/step_errors_all_scales.csv

Run from the repo root:
    python scripts/run_backtest.py
"""
from __future__ import annotations

import logging
import sys

import torch

# Keep statsforecast / prophet / cmdstanpy from spamming stdout.
for name in ("cmdstanpy", "prophet", "statsforecast"):
    logging.getLogger(name).setLevel(logging.ERROR)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    Naive,
    RandomWalkWithDrift,
    Theta,
)

from coffee_forecast import (
    GARCHWrapper,
    GraniteWrapper,
    MLRecursiveWrapper,
    ProphetWrapper,
    StatsForecastWrapper,
    load_coffee_data,
    run_multi_scale_backtest,
)
from coffee_forecast.backtest import step_errors_to_long_df
from coffee_forecast.config import CONTEXT_LEN, CSV_DIR, HORIZON, SCALES, SEED


def build_models() -> dict:
    """Assemble the 10-model suite used in the paper."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading IBM Granite TTM on {device.upper()}...")

    from tsfm_public.toolkit.get_model import get_model
    ttm = (
        get_model(
            model_path="ibm-granite/granite-timeseries-ttm-r2",
            model_name="ttm",
            context_length=CONTEXT_LEN,
            prediction_length=96,
            prefer_longer_context=True,
            return_model_key=False,
        )
        .to(device)
        .eval()
    )

    return {
        # Simple baselines
        "Naive":       StatsForecastWrapper(Naive(alias="Naive")),
        "RWD":         StatsForecastWrapper(RandomWalkWithDrift(alias="RWD")),
        "Theta":       StatsForecastWrapper(Theta(season_length=252, alias="Theta")),
        # Classical statistical
        "ETS":         StatsForecastWrapper(AutoETS(alias="ETS")),
        "ARIMA":       StatsForecastWrapper(AutoARIMA(seasonal=False, alias="ARIMA")),
        # Volatility
        "GARCH":       GARCHWrapper(),
        # Structural decomposition
        "Prophet":     ProphetWrapper(),
        # Recursive ML
        "Ridge":       MLRecursiveWrapper(make_pipeline(StandardScaler(), Ridge())),
        "RF":          MLRecursiveWrapper(
                           RandomForestRegressor(n_estimators=100, random_state=SEED)
                       ),
        # Foundation model
        "Granite-TTM": GraniteWrapper(ttm, device, context_len=CONTEXT_LEN),
    }


def main() -> int:
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    df = load_coffee_data()
    print(
        f"Loaded {len(df)} trading days "
        f"({df['ds'].min().date()} to {df['ds'].max().date()})"
    )

    models = build_models()
    print(f"{len(models)} models initialized.")

    summary, step_errors = run_multi_scale_backtest(
        df=df,
        models=models,
        scales=SCALES,
        context_len=CONTEXT_LEN,
        horizon=HORIZON,
        verbose=True,
    )

    summary_path = CSV_DIR / "summary_all_scales.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    step_errors_df = step_errors_to_long_df(step_errors)
    step_errors_path = CSV_DIR / "step_errors_all_scales.csv"
    step_errors_df.to_csv(step_errors_path, index=False)
    print(f"Saved: {step_errors_path}")

    print(f"\nBacktest complete: {len(summary)} rows across {len(SCALES)} scales.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
