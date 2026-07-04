"""Naive baseline models for honest benchmarking."""
from __future__ import annotations

import numpy as np

from mortality.models.base import MortalityModel


class RandomWalkDrift(MortalityModel):
    """Age-specific random walk with drift on log m(x,t)."""

    name = "random_walk"

    def __init__(self) -> None:
        self.drifts: np.ndarray | None = None
        self.sigmas: np.ndarray | None = None
        self.last_log_mx: np.ndarray | None = None

    def fit(
        self,
        log_mx: np.ndarray,
        ages: np.ndarray,
        years: np.ndarray,
        exposures: np.ndarray | None = None,
        deaths: np.ndarray | None = None,
    ) -> None:
        diffs = np.diff(log_mx, axis=1)
        self.drifts = diffs.mean(axis=1)
        self.sigmas = diffs.std(axis=1, ddof=1)
        self.sigmas = np.where(self.sigmas < 1e-8, 1e-8, self.sigmas)
        self.last_log_mx = log_mx[:, -1]

    def forecast(self, h: int) -> np.ndarray:
        steps = np.arange(1, h + 1)
        return self.last_log_mx[:, None] + self.drifts[:, None] * steps[None, :]

    def simulate(self, h: int, n_paths: int = 1000, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n_ages = len(self.drifts)
        paths = np.zeros((n_paths, n_ages, h))
        for x in range(n_ages):
            innov = rng.normal(0, self.sigmas[x], size=(n_paths, h))
            paths[:, x, 0] = self.last_log_mx[x] + self.drifts[x] + innov[:, 0]
            for t in range(1, h):
                paths[:, x, t] = paths[:, x, t - 1] + self.drifts[x] + innov[:, t]
        return paths


class FrozenRates(MortalityModel):
    """Frozen rates baseline — mortality stays at last observed year."""

    name = "frozen_rates"

    def __init__(self) -> None:
        self.last_log_mx: np.ndarray | None = None

    def fit(
        self,
        log_mx: np.ndarray,
        ages: np.ndarray,
        years: np.ndarray,
        exposures: np.ndarray | None = None,
        deaths: np.ndarray | None = None,
    ) -> None:
        self.last_log_mx = log_mx[:, -1]

    def forecast(self, h: int) -> np.ndarray:
        return np.tile(self.last_log_mx[:, None], (1, h))

    def simulate(self, h: int, n_paths: int = 1000, seed: int = 42) -> np.ndarray:
        fc = self.forecast(h)
        return np.tile(fc[None, :, :], (n_paths, 1, 1))
