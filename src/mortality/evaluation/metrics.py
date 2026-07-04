from __future__ import annotations

import numpy as np

from mortality.actuarial.life_table import annuity_due, life_expectancy_at


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_log_mx(log_mx_true: np.ndarray, log_mx_pred: np.ndarray) -> float:
    return rmse(log_mx_true, log_mx_pred)


def mae_log_mx(log_mx_true: np.ndarray, log_mx_pred: np.ndarray) -> float:
    return mae(log_mx_true, log_mx_pred)


def rmse_e0(mx_true_series: np.ndarray, mx_pred_series: np.ndarray) -> float:
    """RMSE of life expectancy at birth across forecast years.
    Inputs: (n_ages x n_years) matrices of mx.
    """
    n_years = mx_true_series.shape[1]
    e0_true = np.array([life_expectancy_at(mx_true_series[:, t]) for t in range(n_years)])
    e0_pred = np.array([life_expectancy_at(mx_pred_series[:, t]) for t in range(n_years)])
    return rmse(e0_true, e0_pred)


def rmse_annuity_65(
    mx_true_series: np.ndarray,
    mx_pred_series: np.ndarray,
    interest_rate: float = 0.02,
) -> float:
    """RMSE of whole-life annuity-due at 65 across forecast years."""
    n_years = mx_true_series.shape[1]
    a_true = np.array(
        [annuity_due(mx_true_series[:, t], 65, interest_rate) for t in range(n_years)]
    )
    a_pred = np.array(
        [annuity_due(mx_pred_series[:, t], 65, interest_rate) for t in range(n_years)]
    )
    return rmse(a_true, a_pred)


METRIC_REGISTRY = {
    "rmse_log_mx": rmse_log_mx,
    "mae_log_mx": mae_log_mx,
}

ACTUARIAL_METRIC_REGISTRY = {
    "rmse_e0": rmse_e0,
    "rmse_annuity_65": rmse_annuity_65,
}
