"""Booth-Maindonald-Smith model (Booth et al., 2002).

Selects the fitting period by testing linearity of kt.
Adjusts kt to match observed deaths.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from mortality.models.classical.lee_carter import LeeCarter


class BoothMaindonaldSmith(LeeCarter):
    name = "bms"

    def __init__(self, linearity_alpha: float = 0.05) -> None:
        super().__init__()
        self.linearity_alpha = linearity_alpha
        self.fit_start_idx: int = 0

    def fit(
        self,
        log_mx: np.ndarray,
        ages: np.ndarray,
        years: np.ndarray,
        exposures: np.ndarray | None = None,
        deaths: np.ndarray | None = None,
    ) -> None:
        # First fit the standard LC on the full period
        super().fit(log_mx, ages, years, exposures, deaths)

        # Find the longest recent period where kt is linear
        best_start = 0
        n = len(self.kt)

        for start in range(n - 10):
            kt_sub = self.kt[start:]
            t_sub = np.arange(len(kt_sub))
            _, _, _, p_value, _ = stats.linregress(t_sub, kt_sub)
            if p_value < self.linearity_alpha:
                continue
            if n - start > n - best_start:
                best_start = start

        self.fit_start_idx = best_start

        # Refit on the selected period
        if best_start > 0:
            log_mx_sub = log_mx[:, best_start:]
            years_sub = years[best_start:]
            exp_sub = exposures[:, best_start:] if exposures is not None else None
            dth_sub = deaths[:, best_start:] if deaths is not None else None
            super().fit(log_mx_sub, ages, years_sub, exp_sub, dth_sub)
