"""Diebold-Mariano test for pairwise model comparison."""
from __future__ import annotations

import numpy as np
from scipy import stats


def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    loss: str = "squared",
) -> tuple[float, float]:
    """Two-sided Diebold-Mariano test.

    Args:
        e1: forecast errors from model 1 (1D array)
        e2: forecast errors from model 2 (1D array)
        loss: "squared" or "absolute"

    Returns:
        (test_statistic, p_value)
    """
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
    d_var = d.var(ddof=1)

    if d_var < 1e-15:
        return 0.0, 1.0

    dm_stat = d_mean / np.sqrt(d_var / n)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))

    return float(dm_stat), float(p_value)
