"""
Run the daily live forecast. Writes:
    forecasts/latest_forecast.csv
    forecasts/latest_forecast.png

The actual forecasting logic lives in ``coffee_forecast.deployment``;
this script is a thin wrapper for CI / cron use.

Usage:
    python scripts/run_forecast.py
"""
from __future__ import annotations

import sys

from coffee_forecast import load_coffee_data, forecast, fetch_latest, plot_forecast
from coffee_forecast.config import FORECASTS_DIR, REPO_ROOT


def main() -> int:
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the committed history and append any newer closes from Yahoo.
    prices = load_coffee_data()
    prices = fetch_latest(prices)

    # 63 business days ahead with 50/80/95% intervals (deployment defaults).
    fc = forecast(prices, horizon=63)

    csv_path = FORECASTS_DIR / "latest_forecast.csv"
    png_path = FORECASTS_DIR / "latest_forecast.png"
    fc.to_csv(csv_path, index=False)

    # --- Archive: per-run CSV snapshot --------------------------------------
    archive_csv_dir = FORECASTS_DIR / "archive"
    archive_csv_dir.mkdir(parents=True, exist_ok=True)
    run_date = str(fc["run_date"].iloc[0])
    archive_csv_path = archive_csv_dir / f"{run_date}.csv"
    fc.to_csv(archive_csv_path, index=False)

    plot_forecast(prices, fc, png_path)

    p0       = float(prices["y"].iloc[-1])
    print(f"Last observed value: {run_date}")
    print(f"Last close:          {p0:.2f} cents/lb")
    print(
        f"95% PI @ day 63:    "
        f"[{fc['lo_95'].iloc[-1]:.2f}, {fc['hi_95'].iloc[-1]:.2f}]"
    )
    print(f"Wrote: {csv_path.relative_to(REPO_ROOT)}")
    print(f"Wrote: {png_path.relative_to(REPO_ROOT)}")
    print(f"Wrote: {archive_csv_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
