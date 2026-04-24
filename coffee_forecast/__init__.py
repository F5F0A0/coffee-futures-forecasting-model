"""
coffee_forecast
===============

A forecasting research package for daily ICE Coffee C futures prices.

Public API
----------
- load_coffee_data       : load the price series as a clean DataFrame
- calculate_metrics      : MAE / RMSE / MAPE / sMAPE / MASE for one forecast
- get_forecast_origins   : evenly-spaced forecast origins on the series
- run_test               : single-model rolling-window backtest
- run_multi_scale_backtest : loops run_test across scales and models
- Model wrappers         : GraniteWrapper, StatsForecastWrapper,
                           MLRecursiveWrapper, ProphetWrapper, GARCHWrapper
"""

from .data import load_coffee_data
from .metrics import calculate_metrics
from .backtest import get_forecast_origins, run_test, run_multi_scale_backtest
from .models import (
    ModelWrapper,
    GraniteWrapper,
    StatsForecastWrapper,
    MLRecursiveWrapper,
    ProphetWrapper,
    GARCHWrapper,
)
from .forecastability import (
    calculate_spectral_predictability,
    calculate_permutation_entropy,
    calculate_hurst_exponent,
)
from .stats_tests import diebold_mariano_test, model_confidence_set
from .deployment import std_t_ppf, forecast, fetch_latest, plot_forecast

__all__ = [
    # data + metrics
    "load_coffee_data",
    "calculate_metrics",
    # backtest
    "get_forecast_origins",
    "run_test",
    "run_multi_scale_backtest",
    # model wrappers
    "ModelWrapper",
    "GraniteWrapper",
    "StatsForecastWrapper",
    "MLRecursiveWrapper",
    "ProphetWrapper",
    "GARCHWrapper",
    # forecastability
    "calculate_spectral_predictability",
    "calculate_permutation_entropy",
    "calculate_hurst_exponent",
    # statistical tests
    "diebold_mariano_test",
    "model_confidence_set",
    # deployment
    "std_t_ppf",
    "forecast",
    "fetch_latest",
    "plot_forecast",
]

__version__ = "0.1.0"
