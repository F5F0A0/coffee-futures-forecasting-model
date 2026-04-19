"""Run the coffee-futures forecast. Writes a CSV and a PNG.

Usage: python run_forecast.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_utils import load_coffee_data
from forecast import forecast

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "coffee.csv"
OUT_DIR = ROOT / "forecasts"


def fetch_latest(prices: pd.DataFrame) -> pd.DataFrame:
    """Append any newer KC=F closes from Yahoo. In-memory only, no disk writes.

    On any failure, returns the input unchanged with a printed warning —
    we'd rather forecast from slightly-stale data than crash.
    """
    import io
    import logging
    from contextlib import redirect_stderr, redirect_stdout

    next_day = (prices["ds"].max() + pd.Timedelta(days=1)).date().isoformat()

    # Suppress yfinance's chatty error output — we'll print our own.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    sink = io.StringIO()

    try:
        with redirect_stdout(sink), redirect_stderr(sink):
            raw = yf.download(
                "KC=F", start=next_day, progress=False, auto_adjust=False, actions=False
            )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.empty:
            print(f"[fetch] no new rows from Yahoo since {prices['ds'].max().date()}")
            return prices

        col = "Close" if "Close" in raw.columns else "Adj Close"
        new = raw[[col]].reset_index().rename(columns={"Date": "ds", col: "y"})
        new["ds"] = pd.to_datetime(new["ds"]).dt.tz_localize(None).dt.normalize()
        # Yahoo's KC=F is already in cents/lb on the same scale as our CSV.
        # Do NOT multiply by 100.
        new["y"] = new["y"].astype(float)
        new = new.dropna(subset=["y"])
        new = new[new["ds"] >= pd.Timestamp(next_day)]
        if new.empty:
            print(f"[fetch] no new rows from Yahoo since {prices['ds'].max().date()}")
            return prices

        combined = (
            pd.concat([prices, new[["ds", "y"]].assign(unique_id="coffee")], ignore_index=True)
            .sort_values("ds")
            .drop_duplicates(subset="ds", keep="last")
            .reset_index(drop=True)
        )
        print(
            f"[fetch] appended {len(new)} rows from Yahoo "
            f"({new['ds'].min().date()} → {new['ds'].max().date()})"
        )
        return combined
    except Exception as e:
        print(f"[fetch] Yahoo fetch failed ({type(e).__name__}: {e}); using existing history")
        return prices


def plot_forecast(prices: pd.DataFrame, fc: pd.DataFrame, out_path: Path) -> None:
    """Chart: trailing 6 months of history + point forecast + 80/95% PIs."""
    run_date = pd.Timestamp(fc["run_date"].iloc[0])
    trail = prices[prices["ds"] >= run_date - pd.DateOffset(months=6)]
    fc_dates = pd.to_datetime(fc["target_date"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trail["ds"], trail["y"], color="black", lw=1.2, label="Historical prices")
    ax.plot(
        fc_dates, fc["point"], color="steelblue", ls="--", lw=1.5, label="Point forecast (Naïve)"
    )
    ax.fill_between(
        fc_dates,
        fc["lo_95"],
        fc["hi_95"],
        color="steelblue",
        alpha=0.12,
        label="95% prediction interval",
    )
    ax.fill_between(
        fc_dates,
        fc["lo_80"],
        fc["hi_80"],
        color="steelblue",
        alpha=0.28,
        label="80% prediction interval",
    )
    ax.axvline(run_date, color="gray", ls=":", alpha=0.6)
    ax.text(
        run_date, ax.get_ylim()[1] * 0.98, "  Last observed value", fontsize=9, color="gray", va="top"
    )
    ax.set_title(f"Live Coffee Futures Forecast — {run_date.date()}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (cents/lb)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    prices = load_coffee_data(str(DATA_CSV))
    prices = fetch_latest(prices)
    fc = forecast(prices, horizon=63)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_date = fc["run_date"].iloc[0]
    csv_path = OUT_DIR / "latest_forecast.csv"
    png_path = OUT_DIR / "latest_forecast.png"

    fc.to_csv(csv_path, index=False)
    plot_forecast(prices, fc, png_path)

    p0 = float(prices["y"].iloc[-1])
    print(f"Last observed value: {run_date}")
    print(f"Last close:          {p0:.2f} cents/lb")
    print(f"95% PI @ day 63:    [{fc['lo_95'].iloc[-1]:.2f}, {fc['hi_95'].iloc[-1]:.2f}]")
    print(f"Wrote: {csv_path.relative_to(ROOT)}")
    print(f"Wrote: {png_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
