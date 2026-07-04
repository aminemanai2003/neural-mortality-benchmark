"""Poisson Lee-Carter model (Brouhns, Denuit & Vermunt, 2002).

Maximum likelihood estimation under Poisson assumption:
    D(x,t) ~ Poisson(E(x,t) * exp(ax + bx*kt))

Estimated via iterative alternating Newton-Raphson.
"""
from __future__ import annotations

import numpy as np

from mortality.models.base import MortalityModel


class PoissonLeeCarter(MortalityModel):
    name = "poisson_lc"

    def __init__(self, max_iter: int = 500, tol: float = 1e-6) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.ax: np.ndarray | None = None
        self.bx: np.ndarray | None = None
        self.kt: np.ndarray | None = None
        self.drift: float = 0.0
        self.sigma_rwd: float = 0.0

    def fit(
        self,
        log_mx: np.ndarray,
        ages: np.ndarray,
        years: np.ndarray,
        exposures: np.ndarray | None = None,
        deaths: np.ndarray | None = None,
    ) -> None:
        n_ages, n_years = log_mx.shape

        if deaths is None or exposures is None:
            mx = np.exp(log_mx)
            if exposures is None:
                exposures = np.ones_like(mx)
            deaths = mx * exposures

        # Initialize from SVD solution
        self.ax = log_mx.mean(axis=1)
        centered = log_mx - self.ax[:, None]
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        bx = U[:, 0]
        kt = S[0] * Vt[0, :]
        bx_sum = bx.sum()
        bx = bx / bx_sum
        kt = kt * bx_sum

        # Iterative Newton-Raphson
        for iteration in range(self.max_iter):
            ax_old, bx_old, kt_old = self.ax.copy(), bx.copy(), kt.copy()

            # Update ax
            for x in range(n_ages):
                mu = np.exp(self.ax[x] + bx[x] * kt) * exposures[x, :]
                self.ax[x] += (deaths[x, :].sum() - mu.sum()) / mu.sum()

            # Update kt
            for t in range(n_years):
                mu = np.exp(self.ax + bx * kt[t]) * exposures[:, t]
                num = (deaths[:, t] - mu) * bx
                den = mu * bx**2
                if den.sum() != 0:
                    kt[t] += num.sum() / den.sum()

            # Update bx
            for x in range(n_ages):
                mu = np.exp(self.ax[x] + bx[x] * kt) * exposures[x, :]
                num = ((deaths[x, :] - mu) * kt).sum()
                den = (mu * kt**2).sum()
                if den != 0:
                    bx[x] += num / den

            # Identifiability
            kt -= kt.mean()
            bx_s = bx.sum()
            if abs(bx_s) > 1e-10:
                kt *= bx_s
                bx /= bx_s

            # Convergence
            delta = (
                np.max(np.abs(self.ax - ax_old))
                + np.max(np.abs(bx - bx_old))
                + np.max(np.abs(kt - kt_old))
            )
            if delta < self.tol:
                break

        self.bx = bx
        self.kt = kt

        dkt = np.diff(self.kt)
        self.drift = dkt.mean()
        self.sigma_rwd = dkt.std(ddof=1) if len(dkt) > 1 else 0.01

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
        return self.ax[None, :, None] + self.bx[None, :, None] * kt_paths[:, None, :]
