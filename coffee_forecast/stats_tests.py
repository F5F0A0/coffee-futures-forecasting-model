"""
Rigorous model-comparison tests applied to per-origin loss series.

Two complementary tools:

  - Diebold-Mariano (1995) : pairwise test for equal predictive accuracy,
    with a Newey-West HAC variance estimate.

  - Model Confidence Set (Hansen, Lunde & Nason, 2011) : joint procedure
    that returns the smallest set of models which is statistically
    indistinguishable from the best at level alpha.

Both consume the per-origin loss differentials the backtest exports in
``step_errors_all_scales.csv``.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Diebold-Mariano (1995)
# ---------------------------------------------------------------------------
def diebold_mariano_test(
    losses_a,
    losses_b,
    h: int = 1,
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Compares two per-origin loss series (e.g., per-window MAE) and tests
    whether their expected losses are equal. Uses a Newey-West HAC
    variance estimator with Andrews' (1991) automatic bandwidth, with a
    minimum of ``h - 1`` lags to handle the MA(h-1) dependence induced by
    h-step-ahead forecasting.

    Parameters
    ----------
    losses_a, losses_b : array-like
        Per-origin losses for models A and B (same length).
    h : int
        Forecast horizon (default 1). Sets the minimum HAC lag at h-1.

    Returns
    -------
    dm_stat : float
        DM test statistic. Positive => A is worse than B. Negative => A better.
    p_value : float
        Two-sided p-value under H0 of equal accuracy, DM ~ N(0, 1).
    """
    d     = np.asarray(losses_a, float) - np.asarray(losses_b, float)
    T     = len(d)
    d_bar = d.mean()

    # Newey-West HAC variance of the mean of d.
    max_lag = max(h - 1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    x       = d - d_bar
    V       = float(np.dot(x, x)) / T                        # gamma_0
    for k in range(1, max_lag + 1):
        w       = 1.0 - k / (max_lag + 1)                    # Bartlett kernel
        gamma_k = float(np.dot(x[k:], x[:-k])) / T
        V      += 2.0 * w * gamma_k

    se      = np.sqrt(max(V, 1e-12) / T)
    dm_stat = d_bar / se
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))

    return float(dm_stat), float(p_value)


# ---------------------------------------------------------------------------
# Model Confidence Set (Hansen, Lunde & Nason 2011)
# ---------------------------------------------------------------------------
def model_confidence_set(
    losses_dict: Dict[str, np.ndarray],
    alpha: float = 0.10,
    block_size: int = 5,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Model Confidence Set via sequential elimination with circular block
    bootstrap.

    Reference
    ---------
    Hansen, Lunde & Nason (2011), "The Model Confidence Set",
    Econometrica 79(2), 453-497.

    Procedure
    ---------
    1. Start with M = all models.
    2. Build loss differentials d_i,t = L_i,t - mean_j L_j,t over the
       current active set and compute the max-t statistic T_R.
    3. Obtain a null p-value via circular block bootstrap, centering the
       resampled differentials at 0 to simulate H0 (equal accuracy).
    4. If p <= alpha, eliminate the model with the largest t_i and record
       its p-value. Otherwise stop.
    5. Repeat until no model can be rejected.

    Models surviving at level alpha are jointly indistinguishable from
    the best model at confidence (1 - alpha).

    Parameters
    ----------
    losses_dict : dict
        ``{model_name: per_origin_losses}`` - all entries must have the
        same length.
    alpha : float
        Significance level for elimination (default 0.10 -> 90% MCS).
    block_size : int
        Length of each resampled block (default 5).
    n_boot : int
        Number of bootstrap replications (default 1000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    mcs_models : list of str
        Models surviving in the MCS (statistically best-tier).
    results_df : pd.DataFrame
        Columns ``model, mcs_p_value, in_mcs``. Surviving models have
        ``mcs_p_value == 1.0``; eliminated models have the p-value at
        which they were rejected.
    """
    rng   = np.random.default_rng(seed)
    names = list(losses_dict.keys())
    L     = np.column_stack([np.asarray(losses_dict[n], float) for n in names])
    T     = L.shape[0]

    def _nw_se(x: np.ndarray) -> float:
        """Newey-West standard error of the mean of x."""
        T_  = len(x)
        xc  = x - x.mean()
        V   = float(np.dot(xc, xc)) / T_
        mlag = max(1, int(np.floor(4 * (T_ / 100) ** (2 / 9))))
        for k in range(1, mlag + 1):
            w  = 1.0 - k / (mlag + 1)
            V += 2.0 * w * float(np.dot(xc[k:], xc[:-k])) / T_
        return np.sqrt(max(V, 1e-12) / T_)

    def _t_stats(D: np.ndarray) -> np.ndarray:
        """Individual t-statistics for each column of the loss-diff matrix."""
        d_bar = D.mean(axis=0)
        return np.array([d_bar[j] / _nw_se(D[:, j]) for j in range(D.shape[1])])

    def _bootstrap_block(D: np.ndarray) -> np.ndarray:
        """One circular block-bootstrap resample of D (shape T x k)."""
        n_blocks = int(np.ceil(T / block_size))
        starts   = rng.integers(0, T, size=n_blocks)
        idx      = np.concatenate([
            np.arange(s, s + block_size) % T for s in starts
        ])[:T]
        return D[idx]

    active: List[int]        = list(range(len(names)))
    p_values: Dict[str, float] = {}

    for _ in range(len(names) - 1):
        if len(active) <= 1:
            break

        L_sub = L[:, active]
        D     = L_sub - L_sub.mean(axis=1, keepdims=True)
        t_obs = _t_stats(D)
        T_obs = float(t_obs.max())

        # Bootstrap the null distribution of max-t under H0.
        d_bar_obs  = D.mean(axis=0)                    # observed column means
        boot_max_t = np.empty(n_boot)
        for b in range(n_boot):
            D_b           = _bootstrap_block(D)
            D_b_h0        = D_b - d_bar_obs            # center under H0
            boot_max_t[b] = float(_t_stats(D_b_h0).max())

        p_val = float(np.mean(boot_max_t >= T_obs))

        if p_val <= alpha:
            worst_local                  = int(np.argmax(t_obs))
            worst_global                 = active[worst_local]
            p_values[names[worst_global]] = p_val
            active.remove(worst_global)
        else:
            break

    mcs_models = [names[i] for i in active]

    results_df = pd.DataFrame([
        {"model": n,
         "mcs_p_value": p_values.get(n, 1.0),
         "in_mcs": n in mcs_models}
        for n in names
    ]).sort_values("mcs_p_value", ascending=False).reset_index(drop=True)

    return mcs_models, results_df
