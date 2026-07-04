"""Original Lee-Carter model (Lee & Carter, 1992).

SVD decomposition of centered log mortality matrix.
kt re-estimated to match total observed deaths.
kt forecast via random walk with drift.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from mortality.models.base import MortalityModel


class LeeCarter(MortalityModel):
    name = "lee_carter"

    def __init__(self) -> None:
        self.ax: np.ndarray | None = None
        self.bx: np.ndarray | None = None
        self.kt: np.ndarray | None = None
        self.drift: float = 0.0
        self.sigma_rwd: float = 0.0
        self._ages: np.ndarray | None = None
        self._years: np.ndarray | None = None

    def fit(
        self,
        log_mx: np.ndarray,
        ages: np.ndarray,
        years: np.ndarray,
        exposures: np.ndarray | None = None,
        deaths: np.ndarray | None = None,
    ) -> None:
        n_ages, n_years = log_mx.shape
        self._ages = ages
        self._years = years

        # Step 1: ax = row means
        self.ax = log_mx.mean(axis=1)

        # Step 2: SVD on centered matrix
        centered = log_mx - self.ax[:, None]
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # First component
        bx_raw = U[:, 0]
        kt_raw = S[0] * Vt[0, :]

        # Identifiability: sum(bx) = 1, sum(kt) = 0
        self.bx = bx_raw / bx_raw.sum()
        self.kt = kt_raw * bx_raw.sum()

        # Step 3: Re-estimate kt to match observed deaths
        if deaths is not None and exposures is not None:
            self.kt = self._reestimate_kt(deaths, exposures)

        # Step 4: Fit RWD to kt
        dkt = np.diff(self.kt)
        self.drift = dkt.mean()
        self.sigma_rwd = dkt.std(ddof=1)

    def _reestimate_kt(self, deaths: np.ndarray, exposures: np.ndarray) -> np.ndarray:
        """Re-estimate kt so that predicted deaths match observed total deaths per year."""
        n_years = deaths.shape[1]
        kt_new = np.zeros(n_years)

        for t in range(n_years):
            D_obs = deaths[:, t].sum()

            def objective(k: float) -> float:
                log_rates = self.ax + self.bx * k
                D_pred = (np.exp(log_rates) * exposures[:, t]).sum()
                return (D_pred - D_obs) ** 2

            result = minimize_scalar(objective, bounds=(-500, 500), method="bounded")
            kt_new[t] = result.x

        return kt_new

    def forecast(self, h: int) -> np.ndarray:
        kt_last = self.kt[-1]
        kt_fc = kt_last + self.drift * np.arange(1, h + 1)
        return self.ax[:, None] + self.bx[:, None] * kt_fc[None, :]

    def simulate(self, h: int, n_paths: int = 1000, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        kt_last = self.kt[-1]
        innovations = rng.normal(0, self.sigma_rwd, size=(n_paths, h))
        kt_paths = np.zeros((n_paths, h))
        kt_paths[:, 0] = kt_last + self.drift + innovations[:, 0]
        for t in range(1, h):
            kt_paths[:, t] = kt_paths[:, t - 1] + self.drift + innovations[:, t]

        log_mx_sim = self.ax[None, :, None] + self.bx[None, :, None] * kt_paths[:, None, :]
        return log_mx_sim
