"""Lee-Miller variant (Lee & Miller, 2001).

Adjusts kt so that fitted e0 matches observed e0 each year.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from mortality.actuarial.life_table import life_expectancy_at
from mortality.models.classical.lee_carter import LeeCarter


class LeeMiller(LeeCarter):
    name = "lee_miller"

    def fit(
        self,
        log_mx: np.ndarray,
        ages: np.ndarray,
        years: np.ndarray,
        exposures: np.ndarray | None = None,
        deaths: np.ndarray | None = None,
    ) -> None:
        super().fit(log_mx, ages, years, exposures, deaths)
        self.kt = self._adjust_kt_on_e0(log_mx)

        dkt = np.diff(self.kt)
        self.drift = dkt.mean()
        self.sigma_rwd = dkt.std(ddof=1)

    def _adjust_kt_on_e0(self, log_mx: np.ndarray) -> np.ndarray:
        """Adjust kt so that fitted e0 matches observed e0 each year."""
        n_years = log_mx.shape[1]
        kt_adj = np.zeros(n_years)

        for t in range(n_years):
            e0_obs = life_expectancy_at(np.exp(log_mx[:, t]))

            def residual(k: float) -> float:
                log_rates = self.ax + self.bx * k
                return life_expectancy_at(np.exp(log_rates)) - e0_obs

            try:
                kt_adj[t] = brentq(residual, -500, 500)
            except ValueError:
                kt_adj[t] = self.kt[t]

        return kt_adj
