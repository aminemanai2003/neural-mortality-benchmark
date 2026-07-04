from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class MortalityModel(ABC):
    """Common interface for all mortality models."""

    name: str = "base"
    # Ages the forecast covers. None means "same age grid as passed to fit".
    # Models restricting the range (e.g. CBD on 60+) must set this in fit.
    forecast_ages: np.ndarray | None = None

    @abstractmethod
    def fit(
        self,
        log_mx: np.ndarray,
        ages: np.ndarray,
        years: np.ndarray,
        exposures: np.ndarray | None = None,
        deaths: np.ndarray | None = None,
    ) -> None:
        """Fit the model on a (n_ages x n_years) matrix of log central death rates."""

    @abstractmethod
    def forecast(self, h: int) -> np.ndarray:
        """Forecast h years ahead. Returns (n_ages x h) array of log m(x,t)."""

    @abstractmethod
    def simulate(self, h: int, n_paths: int = 1000, seed: int = 42) -> np.ndarray:
        """Simulate n_paths future paths. Returns (n_paths x n_ages x h)."""

    def forecast_mx(self, h: int) -> np.ndarray:
        return np.exp(self.forecast(h))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
