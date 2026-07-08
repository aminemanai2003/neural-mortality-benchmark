"""Cairns-Blake-Dowd model (CBD, 2006).

Two-factor model for older ages (typically 60+):
    logit q(x,t) = kt1 + kt2 * (x - x_bar)

Standard model used by regulators for longevity risk.
"""
from __future__ import annotations

import numpy as np

from mortality.models.base import MortalityModel


class CairnsBlakeDowd(MortalityModel):
    name = "cbd"

    def __init__(self, age_range: tuple[int, int] = (60, 100)) -> None:
        self.age_range = age_range
        self.kt1: np.ndarray | None = None
        self.kt2: np.ndarray | None = None
        self.x_bar: float = 0.0
        self.drift1: float = 0.0
        self.drift2: float = 0.0
        self.cov_dkt: np.ndarray | None = None
        self._ages: np.ndarray | None = None

    def fit(
        self,
        log_mx: np.ndarray,
        ages: np.ndarray,
        years: np.ndarray,
        exposures: np.ndarray | None = None,
        deaths: np.ndarray | None = None,
    ) -> None:
        mask = (ages >= self.age_range[0]) & (ages <= self.age_range[1])
        if not np.any(mask):
            self._ages = np.array([], dtype=ages.dtype)
            self.forecast_ages = self._ages
            self._empty = True
            return
        self._empty = False
        self._ages = ages[mask]
        self.forecast_ages = self._ages
        mx_sub = np.exp(log_mx[mask, :])

        qx = 1.0 - np.exp(-mx_sub)
        qx = np.clip(qx, 1e-8, 1 - 1e-8)
        logit_qx = np.log(qx / (1 - qx))

        self.x_bar = float(self._ages.mean())
        x_centered = self._ages - self.x_bar

        n_years = logit_qx.shape[1]
        self.kt1 = np.zeros(n_years)
        self.kt2 = np.zeros(n_years)

        for t in range(n_years):
            y = logit_qx[:, t]
            X = np.column_stack([np.ones(len(x_centered)), x_centered])
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            self.kt1[t] = beta[0]
            self.kt2[t] = beta[1]

        # Fit bivariate RWD
        dkt = np.column_stack([np.diff(self.kt1), np.diff(self.kt2)])
        self.drift1 = dkt[:, 0].mean()
        self.drift2 = dkt[:, 1].mean()
        self.cov_dkt = np.cov(dkt.T) if len(dkt) > 1 else np.eye(2) * 0.01

    def forecast(self, h: int) -> np.ndarray:
        if getattr(self, "_empty", False):
            return np.empty((0, h))
        x_centered = self._ages - self.x_bar
        kt1_fc = self.kt1[-1] + self.drift1 * np.arange(1, h + 1)
        kt2_fc = self.kt2[-1] + self.drift2 * np.arange(1, h + 1)

        logit_qx = kt1_fc[None, :] + x_centered[:, None] * kt2_fc[None, :]
        qx = 1.0 / (1.0 + np.exp(-logit_qx))
        mx = -np.log(1.0 - qx)
        return np.log(mx)

    def simulate(self, h: int, n_paths: int = 1000, seed: int = 42) -> np.ndarray:
        if getattr(self, "_empty", False):
            return np.empty((n_paths, 0, h))
        rng = np.random.default_rng(seed)
        x_centered = self._ages - self.x_bar
        drift = np.array([self.drift1, self.drift2])

        innovations = rng.multivariate_normal(np.zeros(2), self.cov_dkt, size=(n_paths, h))
        kt_paths = np.zeros((n_paths, h, 2))
        kt_paths[:, 0, :] = np.array([self.kt1[-1], self.kt2[-1]]) + drift + innovations[:, 0, :]
        for t in range(1, h):
            kt_paths[:, t, :] = kt_paths[:, t - 1, :] + drift + innovations[:, t, :]

        k1 = kt_paths[:, :, 0][:, None, :]
        k2 = kt_paths[:, :, 1][:, None, :]
        logit_qx = k1 + x_centered[None, :, None] * k2
        qx = 1.0 / (1.0 + np.exp(-logit_qx))
        mx = -np.log(1.0 - qx)
        return np.log(mx)
