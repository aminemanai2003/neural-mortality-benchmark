"""LC-ResNet: Lee-Carter + Neural Residual Correction.

The hybrid model:
1. Fits a Poisson Lee-Carter as the interpretable skeleton.
2. Trains a small neural network to learn the structured residuals
   (non-linearities, cohort effects) from the LC fit.
3. Uses a horizon-dependent shrinkage: the neural correction is
   multiplied by exp(-lambda * h), so at long horizons the model
   reverts to the stable LC extrapolation.

This preserves interpretability (ax, bx, kt are available) while
capturing patterns that LC misses at short/medium horizons.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mortality.models.base import MortalityModel
from mortality.models.classical.poisson_lc import PoissonLeeCarter


class _ResidualNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...], dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hs in hidden_sizes:
            layers.extend([nn.Linear(in_dim, hs), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hs
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LCResNet(MortalityModel):
    """Lee-Carter + ResNet hybrid model."""

    name = "lc_resnet"

    def __init__(
        self,
        residual_hidden: tuple[int, ...] = (64, 32),
        shrinkage_lambda: float = 0.1,
        multi_population: bool = False,
        lr: float = 0.001,
        epochs: int = 200,
        patience: int = 20,
        seed: int = 42,
        dropout: float = 0.1,
    ) -> None:
        self.residual_hidden = residual_hidden
        self.shrinkage_lambda = shrinkage_lambda
        self.multi_population = multi_population
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.dropout = dropout

        self.lc = PoissonLeeCarter()
        self.net: _ResidualNet | None = None
        self._n_ages: int = 0
        self._n_years: int = 0
        self._residual_mean: float = 0.0
        self._residual_std: float = 1.0

    def fit(
        self,
        log_mx: np.ndarray,
        ages: np.ndarray,
        years: np.ndarray,
        exposures: np.ndarray | None = None,
        deaths: np.ndarray | None = None,
    ) -> None:
        n_ages, n_years = log_mx.shape
        self._n_ages = n_ages
        self._n_years = n_years

        # Step 1: Fit the LC skeleton
        self.lc.fit(log_mx, ages, years, exposures, deaths)

        # Step 2: Compute LC fitted values and residuals
        lc_fitted = self.lc.ax[:, None] + self.lc.bx[:, None] * self.lc.kt[None, :]
        residuals = log_mx - lc_fitted

        self._residual_mean = residuals.mean()
        self._residual_std = residuals.std()
        if self._residual_std < 1e-8:
            self._residual_std = 1.0
        residuals_norm = (residuals - self._residual_mean) / self._residual_std

        # Step 3: Build features for the neural network
        # Features: normalized age, normalized year, bx(age), kt(year)
        age_norm = (ages - ages.mean()) / (ages.std() + 1e-8)
        year_norm = (years - years.mean()) / (years.std() + 1e-8)
        bx_norm = (self.lc.bx - self.lc.bx.mean()) / (self.lc.bx.std() + 1e-8)
        kt_norm = (self.lc.kt - self.lc.kt.mean()) / (self.lc.kt.std() + 1e-8)

        features = []
        targets = []
        for x_idx in range(n_ages):
            for t_idx in range(n_years):
                feat = [age_norm[x_idx], year_norm[t_idx], bx_norm[x_idx], kt_norm[t_idx]]
                features.append(feat)
                targets.append(residuals_norm[x_idx, t_idx])

        X = torch.tensor(np.array(features), dtype=torch.float32)
        y = torch.tensor(np.array(targets), dtype=torch.float32)

        # Step 4: Train the residual network
        torch.manual_seed(self.seed)
        self.net = _ResidualNet(X.shape[1], self.residual_hidden, self.dropout)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        n = len(X)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed))
        split = max(1, int(0.85 * n))
        train_idx, val_idx = perm[:split], perm[split:]

        best_val = float("inf")
        best_state = None
        wait = 0

        self.net.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self.net(X[train_idx])
            loss = nn.functional.mse_loss(pred, y[train_idx])
            loss.backward()
            optimizer.step()

            self.net.eval()
            with torch.no_grad():
                vp = self.net(X[val_idx])
                vl = nn.functional.mse_loss(vp, y[val_idx]).item()
            self.net.train()

            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state:
            self.net.load_state_dict(best_state)
        self.net.eval()

        # Store normalization params for forecasting
        self._age_mean = ages.mean()
        self._age_std = ages.std() + 1e-8
        self._year_mean = years.mean()
        self._year_std = years.std() + 1e-8
        self._bx_mean = self.lc.bx.mean()
        self._bx_std = self.lc.bx.std() + 1e-8
        self._kt_mean = self.lc.kt.mean()
        self._kt_std = self.lc.kt.std() + 1e-8
        self._ages = ages
        self._years = years

    def _predict_residual(self, ages: np.ndarray, kt_values: np.ndarray) -> np.ndarray:
        """Predict residual correction for given ages and kt values."""
        if self.net is None:
            return np.zeros((len(ages), len(kt_values)))

        age_norm = (ages - self._age_mean) / self._age_std
        bx_norm = (self.lc.bx - self._bx_mean) / self._bx_std

        results = np.zeros((len(ages), len(kt_values)))
        with torch.no_grad():
            for t, kt_val in enumerate(kt_values):
                kt_n = (kt_val - self._kt_mean) / self._kt_std
                year_n = (self._years[-1] + t + 1 - self._year_mean) / self._year_std
                features = []
                for x_idx in range(len(ages)):
                    features.append([age_norm[x_idx], year_n, bx_norm[x_idx], kt_n])
                X = torch.tensor(np.array(features), dtype=torch.float32)
                pred = self.net(X).numpy()
                results[:, t] = pred * self._residual_std + self._residual_mean

        return results

    def forecast(self, h: int) -> np.ndarray:
        # LC base forecast
        lc_fc = self.lc.forecast(h)

        # Neural residual correction with horizon shrinkage
        kt_fc = self.lc.kt[-1] + self.lc.drift * np.arange(1, h + 1)
        residual = self._predict_residual(self._ages, kt_fc)

        # Shrinkage: correction decays with horizon
        shrinkage = np.exp(-self.shrinkage_lambda * np.arange(1, h + 1))
        correction = residual * shrinkage[None, :]

        return lc_fc + correction

    def simulate(self, h: int, n_paths: int = 1000, seed: int = 42) -> np.ndarray:
        # Use LC simulation paths + fixed residual correction
        lc_sim = self.lc.simulate(h, n_paths, seed)

        kt_fc = self.lc.kt[-1] + self.lc.drift * np.arange(1, h + 1)
        residual = self._predict_residual(self._ages, kt_fc)
        shrinkage = np.exp(-self.shrinkage_lambda * np.arange(1, h + 1))
        correction = residual * shrinkage[None, :]

        return lc_sim + correction[None, :, :]
