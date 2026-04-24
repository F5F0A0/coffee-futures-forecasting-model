"""
Project-wide configuration constants.

Centralizing these means a reviewer can find every knob in one place,
and downstream notebooks never have to rediscover numbers like 1536.
"""
from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (resolved relative to the repo root, not the caller's CWD)
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT    = PACKAGE_ROOT.parent
DATA_DIR     = REPO_ROOT / "data"
RESULTS_DIR  = REPO_ROOT / "results"
CSV_DIR      = RESULTS_DIR / "csv"
FIG_DIR      = RESULTS_DIR / "figures"
FORECASTS_DIR = REPO_ROOT / "forecasts"   # live-deployment outputs

DEFAULT_DATA_FILE = DATA_DIR / "coffee.csv"

# ---------------------------------------------------------------------------
# Backtest design
# ---------------------------------------------------------------------------
CONTEXT_LEN = 1536   # ~6 trading years of daily history per forecast
HORIZON     = 63     # ~3 trading months ahead
SCALES      = [1, 10, 30, 60]  # evenly-spaced forecast origins

# Single source of truth for reproducibility.
SEED = 42

# ---------------------------------------------------------------------------
# Plotting: colors are consistent across every figure in the paper.
# ---------------------------------------------------------------------------
COLOR_MAP = {
    "Actual":      "#111111",
    "Granite-TTM": "#E69F00",
    "ARIMA":       "#4C72B0",
    "ETS":         "#55A868",
    "Naive":       "#8172B2",
    "RWD":         "#CC79A7",
    "RF":          "#BCBD22",
    "Ridge":       "#8C8C8C",
    "GARCH":       "#17becf",
    "Prophet":     "#d62728",
    "Theta":       "#008080",
}

# Preferred plot / table ordering (leaderboard order from the 60-origin run).
PLOT_ORDER = [
    "RWD", "ETS", "Theta", "Naive", "RF",
    "ARIMA", "GARCH", "Granite-TTM", "Ridge", "Prophet",
]
