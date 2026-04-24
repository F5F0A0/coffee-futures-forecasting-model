"""
A priori forecastability diagnostics.

Three complementary metrics, each capturing a different facet of how
predictable a time series is *before* any model sees it:

  - Spectral Predictability (Omega) : frequency-domain concentration
  - Permutation Entropy             : ordinal-pattern complexity
  - Hurst Exponent (with Lo's test) : long-range dependence in returns

These are the diagnostics that motivate the core claim of the paper: coffee
futures prices behave near-randomly, so simple baselines should be hard to
beat. This module exposes them as standalone callables; the interpretation
and synthesis happens in ``notebooks/02_forecastability.ipynb``.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats
from scipy.signal import detrend


# ---------------------------------------------------------------------------
# Spectral Predictability (Omega)
# ---------------------------------------------------------------------------
def calculate_spectral_predictability(series: np.ndarray) -> float:
    """
    Spectral Predictability (Omega) from Wang & Klee (KDD 2025).

    Omega = 1 - H_spectral / log(N_freq)

    where ``H_spectral`` is the Shannon entropy of the normalized power
    spectral density and ``N_freq = floor(T/2) + 1`` is the number of FFT
    frequency bins. Values closer to 1.0 indicate concentrated periodic
    structure (higher forecastability); below ~0.20 indicates a near-flat
    spectrum (low forecastability).

    We use the discrete-length normalization log(N_freq) rather than the
    paper's continuous-time constant log(2*pi). Paper's normalization
    produces values outside [0, 1] on long discrete series because
    log(N_freq) >> log(2*pi) when T is large. Normalizing by the white-noise
    entropy at the same length keeps Omega in [0, 1] and makes the 0.20
    threshold interpretable regardless of T.

    Returns
    -------
    float
        Omega in [0, 1] under the discrete normalization.
    """
    series = np.asarray(series, dtype=float)

    # Remove linear trend so Omega measures *periodic* structure, not drift.
    detrended = detrend(series)

    # Hann window to mitigate spectral leakage from finite-length effects.
    windowed = detrended * np.hanning(len(detrended))

    # Power spectral density from the real FFT.
    fft_vals = np.fft.rfft(windowed)
    psd      = np.abs(fft_vals) ** 2
    psd_norm = psd / np.sum(psd)

    # Shannon entropy of the PSD (eps prevents log(0)).
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))

    return float(1.0 - spectral_entropy / np.log(len(psd_norm)))


# ---------------------------------------------------------------------------
# Permutation Entropy (thin wrapper around ordpy for documentation)
# ---------------------------------------------------------------------------
def calculate_permutation_entropy(
    series: np.ndarray,
    order: int = 3,
    delay: int = 1,
) -> float:
    """
    Normalized permutation entropy via ``ordpy``.

    Measures complexity through the distribution of ordinal patterns of
    length ``order`` in the series. Output is in [0, 1]:

      - 1.0  : maximally random (all permutations equally likely)
      - 0.0  : fully deterministic (one permutation dominates)

    Low permutation entropy implies an exploitable deterministic component;
    values near 1.0 suggest the series is noise-dominated.

    Parameters
    ----------
    series : array-like
        The time series to analyze.
    order : int
        Embedding dimension (default 3; typical range 3-7).
    delay : int
        Embedding delay (default 1 for daily data).

    Returns
    -------
    float
        Normalized permutation entropy in [0, 1].
    """
    from ordpy import permutation_entropy  # lazy: ordpy is a niche dep

    return float(permutation_entropy(np.asarray(series), dx=order, taux=delay))


# ---------------------------------------------------------------------------
# Hurst Exponent (classical R/S) + Lo (1991) modified R/S test
# ---------------------------------------------------------------------------
def calculate_hurst_exponent(
    series: np.ndarray,
    n_min: int = 10,
    n_points: int = 40,
) -> Tuple[float, float, str]:
    """
    Classical Hurst exponent via rescaled-range (R/S) analysis,
    validated by Lo's (1991) modified R/S test for long memory.

    Computed on log-returns, not raw prices, because R/S analysis requires
    a stationary series. Raw prices are I(1); log-returns are stationary
    under standard assumptions.

    Interpretation (classical H):
      - H = 0.5 : random walk, no exploitable memory
      - H > 0.5 : persistent / trending
      - H < 0.5 : anti-persistent / mean-reverting

    Lo's modified test corrects for short-range autocorrelation using a
    Bartlett-kernel HAC estimator. The test statistic V_n lies in the 95%
    acceptance interval [0.809, 1.862] under H0 (no long memory).

    Parameters
    ----------
    series : array-like
        Price series (log-returns are computed internally).
    n_min : int
        Smallest window size in the R/S log-log regression.
    n_points : int
        Number of geometrically-spaced window sizes used for the fit.

    Returns
    -------
    h : float
        Classical Hurst exponent (OLS slope of the log-log R/S plot).
    v_n : float
        Lo's modified R/S test statistic, V_n = R_tilde_S / sqrt(T).
    lo_result : str
        ``"fail to reject"`` if V_n in [0.809, 1.862], else ``"reject"``.
    """
    series = np.asarray(series, dtype=float)
    log_returns = np.diff(np.log(series))
    T = len(log_returns)

    # Geometric spacing of window sizes for the log-log regression.
    n_max = T // 4
    window_sizes = np.unique(
        np.logspace(np.log10(n_min), np.log10(n_max), n_points).astype(int)
    )

    def _compute_rs(x: np.ndarray, n: int) -> float:
        """Mean R/S statistic over non-overlapping windows of length n."""
        num_windows = len(x) // n
        if num_windows == 0:
            return np.nan
        rs_values = []
        for w in range(num_windows):
            chunk = x[w * n : (w + 1) * n]
            e = chunk - chunk.mean()
            X = np.cumsum(e)
            R = X.max() - X.min()
            S = chunk.std(ddof=1)
            if S > 0:
                rs_values.append(R / S)
        return float(np.mean(rs_values)) if rs_values else np.nan

    rs_means = np.array([_compute_rs(log_returns, n) for n in window_sizes])
    valid    = ~np.isnan(rs_means) & (rs_means > 0)
    h, *_    = stats.linregress(np.log(window_sizes[valid]),
                                np.log(rs_means[valid]))

    # --- Lo (1991) modified R/S test ---
    q       = int(np.floor(T ** (1 / 3)))         # Andrews-style bandwidth
    e       = log_returns - log_returns.mean()
    X       = np.cumsum(e)
    R       = X.max() - X.min()
    gamma_0 = np.mean(e ** 2)
    gammas  = np.array([np.mean(e[j:] * e[:-j]) for j in range(1, q + 1)])
    weights = 1.0 - np.arange(1, q + 1) / (q + 1)  # Bartlett kernel
    sigma_q = np.sqrt(gamma_0 + 2.0 * np.sum(weights * gammas))
    v_n     = (R / sigma_q) / np.sqrt(T)

    lo_result = "fail to reject" if 0.809 <= v_n <= 1.862 else "reject"

    return float(h), float(v_n), lo_result
