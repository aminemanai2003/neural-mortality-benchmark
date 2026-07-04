"""Diebold-Mariano test for pairwise model comparison.

Includes the Harvey-Leybourne-Newbold (1997) small-sample / multi-step correction:
for h-step-ahead forecasts the loss differentials are autocorrelated up to lag h-1,
so the naive DM statistic is anti-conservative. We use the autocovariance-adjusted
variance and a Student-t reference distribution.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    loss: str = "squared",
    horizon: int = 1,
) -> tuple[float, float]:
    """Two-sided Diebold-Mariano test with the HLN correction.

    Args:
        e1: forecast errors from model 1 (1D array)
        e2: forecast errors from model 2 (1D array)
        loss: "squared" or "absolute"
        horizon: forecast horizon h; loss differentials are treated as
            autocorrelated up to lag h-1.

    Returns:
        (test_statistic, p_value). A negative statistic favours model 1.
    """
    e1 = np.asarray(e1, dtype=float).ravel()
    e2 = np.asarray(e2, dtype=float).ravel()

    if loss == "squared":
        d = e1**2 - e2**2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    n = len(d)
    if n < 2:
        return 0.0, 1.0

    d_mean = d.mean()
    h = max(1, int(horizon))

    # Autocovariance-adjusted long-run variance up to lag h-1.
    gamma0 = np.mean((d - d_mean) ** 2)
    var_d = gamma0
    for lag in range(1, min(h, n)):
        gamma = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
        var_d += 2.0 * gamma

    var_d = var_d / n
    if var_d <= 1e-15:
        return 0.0, 1.0

    dm_stat = d_mean / np.sqrt(var_d)

    # Harvey-Leybourne-Newbold small-sample correction factor.
    correction = np.sqrt(
        max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-8)
    )
    dm_stat *= correction

    p_value = 2.0 * (1.0 - stats.t.cdf(abs(dm_stat), df=n - 1))

    return float(dm_stat), float(p_value)
