"""CNN model on the mortality surface.

Treats the (age x year) log-mortality matrix as a 2D image
and uses a CNN to learn temporal patterns for forecasting.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mortality.models.base import MortalityModel


class _CNNNet(nn.Module):
    def __init__(
        self,
        n_ages: int,
        lookback: int,
        channels: tuple[int, ...] = (16, 32, 64),
        kernel_size: int = 3,
    ):
        super().__init__()
        layers = []
        in_ch = 1
        for ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.BatchNorm2d(ch),
            ])
            in_ch = ch
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((n_ages, 1))
        self.fc = nn.Linear(channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_ages, lookback)
        feat = self.cnn(x)
        pooled = self.pool(feat)  # (batch, channels, n_ages, 1)
        pooled = pooled.squeeze(-1).permute(0, 2, 1)  # (batch, n_ages, channels)
        return self.fc(pooled).squeeze(-1)  # (batch, n_ages)


class CNNSurface(MortalityModel):
    name = "cnn_surface"

    def __init__(
        self,
        channels: tuple[int, ...] = (16, 32, 64),
        kernel_size: int = 3,
        lookback: int = 20,
        lr: float = 0.001,
        epochs: int = 200,
        patience: int = 20,
        seed: int = 42,
    ) -> None:
        self.channels = channels
        self.kernel_size = kernel_size
        self.lookback = lookback
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.net: _CNNNet | None = None
        self._log_mx_mean: float = 0.0
        self._log_mx_std: float = 1.0
        self._last_window: np.ndarray | None = None
        self._n_ages: int = 0

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

        self._log_mx_mean = log_mx.mean()
        self._log_mx_std = log_mx.std()
        norm = (log_mx - self._log_mx_mean) / self._log_mx_std

        X, y = [], []
        for t in range(n_years - self.lookback):
            X.append(norm[:, t : t + self.lookback])
            y.append(norm[:, t + self.lookback])

        X = np.array(X)[:, None, :, :]  # (N, 1, n_ages, lookback)
        y = np.array(y)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        split = max(1, int(0.8 * len(X_t)))
        X_train, y_train = X_t[:split], y_t[:split]
        X_val, y_val = X_t[split:], y_t[split:]

        torch.manual_seed(self.seed)
        self.net = _CNNNet(n_ages, self.lookback, self.channels, self.kernel_size)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        best_val = float("inf")
        best_state = None
        wait = 0

        self.net.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self.net(X_train)
            loss = nn.functional.mse_loss(pred, y_train)
            loss.backward()
            optimizer.step()

            if len(X_val) > 0:
                self.net.eval()
                with torch.no_grad():
                    vp = self.net(X_val)
                    vl = nn.functional.mse_loss(vp, y_val).item()
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
        self._last_window = norm[:, -self.lookback:]

    def forecast(self, h: int) -> np.ndarray:
        window = self._last_window.copy()
        results = []

        with torch.no_grad():
            for _ in range(h):
                x = torch.tensor(window[None, None, :, :], dtype=torch.float32)
                pred = self.net(x).numpy()[0]
                results.append(pred)
                window = np.concatenate([window[:, 1:], pred[:, None]], axis=1)

        out = np.column_stack(results)
        return out * self._log_mx_std + self._log_mx_mean

    def simulate(self, h: int, n_paths: int = 1000, seed: int = 42) -> np.ndarray:
        fc = self.forecast(h)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 0.05, size=(n_paths, *fc.shape))
        return fc[None, :, :] + noise
