"""Hyndman-Ullah functional data model (Hyndman & Ullah, 2007).

Functional PCA on smoothed log mortality curves, with ARIMA on the scores.
Simplified here: uses polynomial smoothing + standard PCA + ARIMA(0,1,0) = RWD.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline

from mortality.models.base import MortalityModel


class HyndmanUllah(MortalityModel):
    name = "hyndman_ullah"

    def __init__(self, n_components: int = 6) -> None:
        self.n_components = n_components
        self.mu: np.ndarray | None = None
        self.basis: np.ndarray | None = None  # (n_ages, n_components)
        self.scores: np.ndarray | None = None  # (n_years, n_components)
        self.score_drifts: np.ndarray | None = None
        self.score_sigmas: np.ndarray | None = None
        self._ages: np.ndarray | None = None

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

        # Step 1: Smooth each year's log mx curve
        smoothed = np.zeros_like(log_mx)
        for t in range(n_years):
            try:
                spl = UnivariateSpline(ages, log_mx[:, t], s=n_ages * 0.1)
                smoothed[:, t] = spl(ages)
            except Exception:
                smoothed[:, t] = log_mx[:, t]

        # Step 2: Mean function
        self.mu = smoothed.mean(axis=1)

        # Step 3: PCA on centered smoothed data
        centered = smoothed - self.mu[:, None]
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        k = min(self.n_components, min(n_ages, n_years))
        self.basis = U[:, :k]
        self.scores = (S[:k, None] * Vt[:k, :]).T  # (n_years, k)

        # Step 4: RWD on each score series
        self.score_drifts = np.zeros(k)
        self.score_sigmas = np.zeros(k)
        for j in range(k):
            ds = np.diff(self.scores[:, j])
            self.score_drifts[j] = ds.mean()
            self.score_sigmas[j] = ds.std(ddof=1) if len(ds) > 1 else 0.01

    def forecast(self, h: int) -> np.ndarray:
        k = len(self.score_drifts)
        scores_fc = np.zeros((h, k))
        for j in range(k):
            scores_fc[:, j] = self.scores[-1, j] + self.score_drifts[j] * np.arange(1, h + 1)

        return self.mu[:, None] + self.basis @ scores_fc.T

    def simulate(self, h: int, n_paths: int = 1000, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        k = len(self.score_drifts)

        score_paths = np.zeros((n_paths, h, k))
        for j in range(k):
            innov = rng.normal(0, self.score_sigmas[j], size=(n_paths, h))
            score_paths[:, 0, j] = self.scores[-1, j] + self.score_drifts[j] + innov[:, 0]
            for t in range(1, h):
                score_paths[:, t, j] = (
                    score_paths[:, t - 1, j] + self.score_drifts[j] + innov[:, t]
                )

        # (n_paths, n_ages, h)
        log_mx = self.mu[None, :, None] + np.einsum("ak,ptk->pat", self.basis, score_paths)
        return log_mx
