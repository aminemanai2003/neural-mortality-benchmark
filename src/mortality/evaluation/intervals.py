"""Prediction interval coverage evaluation."""
from __future__ import annotations

import numpy as np


def empirical_coverage(
    log_mx_true: np.ndarray,
    log_mx_simulations: np.ndarray,
    level: float = 0.95,
) -> float:
    """Compute empirical coverage of prediction intervals.

    Args:
        log_mx_true: (n_ages, n_years) actual values
        log_mx_simulations: (n_paths, n_ages, n_years) simulated paths
        level: nominal coverage level

    Returns:
        Fraction of observations within the interval.
    """
    alpha = 1 - level
    lo = np.percentile(log_mx_simulations, 100 * alpha / 2, axis=0)
    hi = np.percentile(log_mx_simulations, 100 * (1 - alpha / 2), axis=0)

    inside = (log_mx_true >= lo) & (log_mx_true <= hi)
    return float(inside.mean())
