"""
Live deployment: GJR-GARCH(1,1)-t forecast with calibrated prediction intervals.

The research repo chooses GJR-GARCH(1,1) with standardized Student-t innovations
as the production model for daily-cadence price-path forecasts. The justification
lives in ``notebooks/05_deployment_garch.ipynb``:

  - simple baselines win on *point* accuracy (notebook 03: MCS contains them all),
    but none of them produce prediction intervals;
  - log-returns exhibit strong volatility clustering (ARCH test, ACF^2), and
    heavy tails (fitted nu ~ 5.6), which a normal GARCH cannot model;
  - GJR adds an asymmetric-leverage term that improves AIC, BIC, and
    Ljung-Box on squared residuals over plain GARCH(1,1)-t;
  - expanding-window backtest at 60 origins gives empirical coverage of
    ~80% at the 80% level and ~97% at the 95% level - well calibrated.

This module just *produces* the forecast. The validation lives in the notebook.
"""
from __future__ import annotations

import io
import logging
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Standardized Student-t quantile (the one subtle bit of the pipeline)
# ---------------------------------------------------------------------------
def std_t_ppf(p: float, nu: float) -> float:
    """
    Quantile of the STANDARDIZED Student-t distribution (variance = 1).

    ``arch_model`` reports variance forecasts in the standardized scale, so
    quantiles used to build prediction intervals must match. ``scipy.stats.t.ppf``
    returns quantiles of the STANDARD Student-t (variance ``nu / (nu - 2)``),
    which inflates intervals by ``sqrt(nu / (nu - 2))``. At ``nu = 5.7`` that's
    about 20% too wide - noticeable on the live chart.

    Parameters
    ----------
    p : float in (0, 1)
        Cumulative probability.
    nu : float > 2
        Degrees of freedom of the Student-t distribution.

    Returns
    -------
    float
        The p-th quantile of the unit-variance Student-t with ``nu`` dof.
    """
    if nu <= 2:
        raise ValueError(f"Student-t variance undefined for nu={nu}")
    return float(stats.t.ppf(p, nu) * np.sqrt((nu - 2.0) / nu))


# ---------------------------------------------------------------------------
# The forecast
# ---------------------------------------------------------------------------
def forecast(
    prices: pd.DataFrame,
    horizon: int = 63,
    levels: Tuple[float, ...] = (0.50, 0.80, 0.95),
) -> pd.DataFrame:
    """
    Fit GJR-GARCH(1,1)-t on log-returns and produce a multi-step price forecast
    with prediction intervals.

    Parameters
    ----------
    prices : DataFrame
        Columns ``ds`` (datetime) and ``y`` (price in cents/lb). Any extra
        columns are ignored.
    horizon : int
        Number of business days ahead to forecast.
    levels : tuple of float in (0, 1)
        Coverage levels for the prediction intervals (e.g. 0.80 -> 80% PI).

    Returns
    -------
    DataFrame
        One row per forecast day. Columns:
        ``run_date, target_date, horizon_days, point, ann_vol``,
        plus ``lo_XX`` / ``hi_XX`` for each level ``XX``.
    """
    from arch import arch_model

    df = prices.sort_values("ds").reset_index(drop=True)

    # arch_model wants returns in percent scale for numerical stability.
    # Tiny parameters confuse the optimizer; multiplying by 100 fixes it.
    r_pct = np.log(df["y"]).diff().dropna().values * 100.0

    # GJR-GARCH(1,1) with Student-t innovations. Constant mean because the
    # drift is statistically indistinguishable from zero (validated in the
    # notebook with HAC-robust standard errors).
    am  = arch_model(r_pct, mean="Constant", vol="Garch",
                     p=1, o=1, q=1, dist="t")
    fit = am.fit(disp="off", show_warning=False)

    # Multi-step variance forecast in percent-squared units.
    var_fc_pct2 = fit.forecast(horizon=horizon).variance.iloc[-1].values
    step_var    = var_fc_pct2 / 10_000.0        # -> decimal squared
    mu_frac     = fit.params["mu"] / 100.0      # per-day drift in log-returns
    nu          = float(fit.params["nu"])

    # Annualized volatility, % (handy summary statistic per-step).
    ann_vol = np.sqrt(step_var * 252.0) * 100.0

    # Cumulative log-return std dev: sum of independent variances under the
    # uncorrelated-returns assumption (ACF on returns was flat in the notebook).
    cum_sd = np.sqrt(np.cumsum(step_var))

    P0 = float(df["y"].iloc[-1])
    run_date = df["ds"].iloc[-1].date()
    horizons = np.arange(1, horizon + 1)

    # Business-day target calendar (US futures trading days).
    target_dates = pd.bdate_range(
        start=pd.Timestamp(run_date) + pd.tseries.offsets.BDay(1),
        periods=horizon,
    )

    drift = mu_frac * horizons
    point = P0 * np.exp(drift)

    out = pd.DataFrame({
        "run_date":     run_date,
        "target_date":  target_dates.date,
        "horizon_days": horizons,
        "point":        point,
        "ann_vol":      ann_vol,
    })

    # Prediction intervals from standardized-t quantiles of the cumulative std dev.
    for lvl in levels:
        q   = std_t_ppf(0.5 + lvl / 2.0, nu)
        pct = int(round(lvl * 100))
        out[f"lo_{pct}"] = P0 * np.exp(drift - q * cum_sd)
        out[f"hi_{pct}"] = P0 * np.exp(drift + q * cum_sd)

    # Round for readability / stable CSV diffs across runs.
    round_cols = ["point", "ann_vol"] + [
        f"{side}_{int(lvl * 100)}" for lvl in levels for side in ("lo", "hi")
    ]
    out[round_cols] = out[round_cols].round(2)
    return out


# ---------------------------------------------------------------------------
# Yahoo Finance catch-up (no disk writes, in-memory only)
# ---------------------------------------------------------------------------
def fetch_latest(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Append any newer ``KC=F`` closes from Yahoo Finance to the loaded series.

    In-memory only - does NOT write back to ``data/coffee.csv``. We prefer
    a slightly-stale forecast to crashing the pipeline when Yahoo is flaky
    (rate limits, DNS hiccups, markets closed, etc.), so any fetch failure
    is logged and the input is returned unchanged.

    Parameters
    ----------
    prices : DataFrame
        Columns ``ds``, ``y``, ``unique_id``.

    Returns
    -------
    DataFrame
        Same schema, with any newer trading days appended.
    """
    import yfinance as yf

    next_day = (prices["ds"].max() + pd.Timedelta(days=1)).date().isoformat()

    # Silence yfinance's chatty stdout/stderr - we print our own one-liners.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    sink = io.StringIO()

    try:
        with redirect_stdout(sink), redirect_stderr(sink):
            raw = yf.download(
                "KC=F", start=next_day, progress=False,
                auto_adjust=False, actions=False,
            )

        # yfinance can return a MultiIndex on columns for multi-ticker queries.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        if raw.empty:
            print(f"[fetch] no new rows since {prices['ds'].max().date()}")
            return prices

        col = "Close" if "Close" in raw.columns else "Adj Close"
        new = (
            raw[[col]]
              .reset_index()
              .rename(columns={"Date": "ds", col: "y"})
        )
        new["ds"] = pd.to_datetime(new["ds"]).dt.tz_localize(None).dt.normalize()
        # Yahoo's KC=F is already in cents/lb on the same scale as our CSV.
        # Do NOT multiply by 100.
        new["y"]  = new["y"].astype(float)
        new       = new.dropna(subset=["y"])
        new       = new[new["ds"] >= pd.Timestamp(next_day)]

        if new.empty:
            print(f"[fetch] no new rows since {prices['ds'].max().date()}")
            return prices

        combined = (
            pd.concat([prices,
                       new[["ds", "y"]].assign(unique_id="coffee")],
                       ignore_index=True)
              .sort_values("ds")
              .drop_duplicates(subset="ds", keep="last")
              .reset_index(drop=True)
        )
        print(
            f"[fetch] appended {len(new)} rows "
            f"({new['ds'].min().date()} -> {new['ds'].max().date()})"
        )
        return combined

    except Exception as e:
        print(f"[fetch] Yahoo fetch failed ({type(e).__name__}: {e}); "
              f"using existing history")
        return prices


# ---------------------------------------------------------------------------
# Plotting the live forecast
# ---------------------------------------------------------------------------
def plot_forecast(
    prices: pd.DataFrame,
    fc: pd.DataFrame,
    out_path: Union[str, Path, None] = None,
    trail_months: int = 6,
) -> None:
    """
    Trailing history + point forecast + 80%/95% PIs, all on one chart.

    Same layout as the live-deployment image in the README so the figure in
    the paper and the figure on GitHub stay in sync.
    """
    import datetime as _dt

    import matplotlib.pyplot as plt

    run_date = pd.Timestamp(fc["run_date"].iloc[0])
    trail    = prices[prices["ds"] >= run_date - pd.DateOffset(months=trail_months)]
    fc_dates = pd.to_datetime(fc["target_date"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trail["ds"], trail["y"], color="black", lw=1.2,
            label="Historical prices")
    ax.plot(fc_dates, fc["point"], color="steelblue", ls="--", lw=1.5,
            label="Point forecast")
    ax.fill_between(fc_dates, fc["lo_95"], fc["hi_95"],
                    color="steelblue", alpha=0.12,
                    label="95% prediction interval")
    ax.fill_between(fc_dates, fc["lo_80"], fc["hi_80"],
                    color="steelblue", alpha=0.28,
                    label="80% prediction interval")
    ax.axvline(run_date, color="gray", ls=":", alpha=0.6)
    ax.text(run_date, ax.get_ylim()[1] * 0.98, "  Last observed value",
            fontsize=9, color="gray", va="top")

    run_stamp = (
        str(fc["run_date"].iloc[0])
        if "run_date" in fc.columns
        else _dt.date.today().isoformat()
    )
    ax.set_title(
        f"Live Coffee Futures Forecast - {run_date.date()}",
        fontsize=13, fontweight="semibold", pad=12,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (cents/lb)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    last_date = prices["ds"].iloc[-1]
    last_price = prices["y"].iloc[-1]
    ax.annotate(
        f"{last_price:.2f}",
        xy=(last_date, last_price),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right", va="bottom",
        fontsize=9, color="black", fontweight="semibold",
    )

    first_fc_date = fc["target_date"].iloc[0]
    first_fc_value = fc["point"].iloc[0]
    ax.annotate(
        f"{first_fc_value:.1f}",
        xy=(first_fc_date, first_fc_value),
        xytext=(8, -10),
        textcoords="offset points",
        ha="left", va="top",
        fontsize=9, color="#1f6091", fontweight="semibold",
    )

    last_fc_date = fc["target_date"].iloc[-1]
    ax.annotate(f"{fc['hi_95'].iloc[-1]:.1f}",
                xy=(last_fc_date, fc['hi_95'].iloc[-1]),
                xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8, color="#888")
    ax.annotate(f"{fc['lo_95'].iloc[-1]:.1f}",
                xy=(last_fc_date, fc['lo_95'].iloc[-1]),
                xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8, color="#888")
    ax.annotate(f"{fc['hi_80'].iloc[-1]:.1f}",
                xy=(last_fc_date, fc['hi_80'].iloc[-1]),
                xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=9, color="#1f6091",
                fontweight="semibold")
    ax.annotate(f"{fc['lo_80'].iloc[-1]:.1f}",
                xy=(last_fc_date, fc['lo_80'].iloc[-1]),
                xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=9, color="#1f6091",
                fontweight="semibold")

    ax.set_xlim(right=last_fc_date + pd.Timedelta(days=10))

    ax.hlines(
        y=last_price,
        xmin=last_date,
        xmax=last_fc_date,
        linestyles="dotted",
        colors="gray",
        linewidth=0.8,
        alpha=0.5,
        zorder=1,
    )

    fig.text(
        0.5, -0.02,
        "GJR-GARCH(1,1)-t with Student-t innovations. Bands show 80% and 95% "
        "probability ranges; empirical coverage is within 3 pp of nominal "
        "across a 60-origin backtest.",
        ha="center", va="top",
        fontsize=8, color="#555", style="italic",
    )

    fig.text(
        0.99, -0.05,
        f"ICE Coffee 'C' (KC=F)  -  Source: Yahoo Finance  -  Run: {run_stamp}",
        ha="right", va="top",
        fontsize=7, color="#999",
    )

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
