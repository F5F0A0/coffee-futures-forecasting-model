"""GJR-GARCH(1,1)-t forecast for coffee futures.

The choice of model, distribution, and interval construction is validated
in notebooks/research_spike_v3.ipynb. This module just produces forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats


def std_t_ppf(p: float, nu: float) -> float:
    """Quantile of the STANDARDIZED Student-t (variance = 1).

    `arch_model` reports variance forecasts in the standardized scale, so
    quantiles used to build prediction intervals must match. `stats.t.ppf`
    returns quantiles of the STANDARD Student-t (variance ν/(ν−2)), which
    inflates intervals by sqrt(ν/(ν−2)). At ν=5.7 that's ~20% too wide.
    """
    if nu <= 2:
        raise ValueError(f"Student-t variance undefined for nu={nu}")
    return float(stats.t.ppf(p, nu) * np.sqrt((nu - 2.0) / nu))


def forecast(
    prices: pd.DataFrame,
    horizon: int = 63,
    levels: tuple[float, ...] = (0.50, 0.80, 0.95),
) -> pd.DataFrame:
    """Fit GJR-GARCH(1,1)-t on log-returns and produce a forecast.

    Parameters
    ----------
    prices : DataFrame with columns [ds, y]; y in cents/lb.
    horizon : business days ahead to forecast.
    levels : prediction interval coverage levels.

    Returns
    -------
    DataFrame with columns:
        run_date, target_date, horizon_days, point, ann_vol,
        lo_50, hi_50, lo_80, hi_80, lo_95, hi_95
    """
    df = prices.sort_values("ds").reset_index(drop=True)
    r_pct = np.log(df["y"]).diff().dropna().values * 100.0

    # Fit GJR-GARCH(1,1) with Student-t innovations. Choice justified in
    # notebooks/research_spike_v3.ipynb §8: beats plain GARCH on AIC/BIC and
    # cleans up Ljung-Box on squared residuals; ties EGARCH/APARCH
    # out-of-sample while retaining analytic multi-step forecasts.
    am = arch_model(r_pct, mean="Constant", vol="Garch",
                    p=1, o=1, q=1, dist="t")
    fit = am.fit(disp="off", show_warning=False)

    # Multi-step variance forecast in %² units
    var_fc_pct2 = fit.forecast(horizon=horizon).variance.iloc[-1].values
    step_var = var_fc_pct2 / 10_000.0          # → decimal²
    mu_frac = fit.params["mu"] / 100.0          # per-day drift in log-returns
    nu = fit.params["nu"]

    ann_vol = np.sqrt(step_var * 252.0) * 100.0       # annualized %
    cum_sd = np.sqrt(np.cumsum(step_var))              # cumulative std dev of log-return

    P0 = float(df["y"].iloc[-1])
    run_date = df["ds"].iloc[-1].date()
    horizons = np.arange(1, horizon + 1)
    target_dates = pd.bdate_range(
        start=pd.Timestamp(run_date) + pd.tseries.offsets.BDay(1),
        periods=horizon,
    )
    drift = mu_frac * horizons
    point = P0 * np.exp(drift)

    out = pd.DataFrame({
        "run_date": run_date,
        "target_date": target_dates.date,
        "horizon_days": horizons,
        "point": point,
        "ann_vol": ann_vol,
    })
    for lvl in levels:
        q = std_t_ppf(0.5 + lvl / 2.0, nu)
        pct = int(round(lvl * 100))
        out[f"lo_{pct}"] = P0 * np.exp(drift - q * cum_sd)
        out[f"hi_{pct}"] = P0 * np.exp(drift + q * cum_sd)

    # Round for readability
    round_cols = ["point", "ann_vol"] + [
        f"{side}_{int(lvl*100)}" for lvl in levels for side in ("lo", "hi")
    ]
    out[round_cols] = out[round_cols].round(2)
    return out
