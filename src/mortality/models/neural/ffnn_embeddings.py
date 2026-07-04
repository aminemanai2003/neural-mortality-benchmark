"""Feed-Forward Neural Network with entity embeddings (Richman & Wüthrich, 2019).

Direct mortality surface model: inputs are (age, year, sex, country) embeddings.
Output is log m(x,t).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mortality.models.base import MortalityModel


class _FFNNNet(nn.Module):
    def __init__(
        self,
        n_ages: int,
        n_years: int,
        embedding_dim: int = 8,
        hidden_sizes: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.age_emb = nn.Embedding(n_ages, embedding_dim)
        self.year_emb = nn.Embedding(n_years, embedding_dim)

        input_dim = embedding_dim * 2
        layers = []
        for hs in hidden_sizes:
            layers.extend([nn.Linear(input_dim, hs), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hs
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, age_idx: torch.Tensor, year_idx: torch.Tensor) -> torch.Tensor:
        a = self.age_emb(age_idx)
        y = self.year_emb(year_idx)
        x = torch.cat([a, y], dim=-1)
        return self.mlp(x).squeeze(-1)


class FFNNEmbeddings(MortalityModel):
    name = "ffnn_embeddings"

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (128, 64, 32),
        embedding_dim: int = 8,
        lr: float = 0.001,
        epochs: int = 200,
        patience: int = 20,
        seed: int = 42,
        dropout: float = 0.1,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.dropout = dropout
        self.net: _FFNNNet | None = None
        self._n_ages: int = 0
        self._n_years: int = 0
        self._ages: np.ndarray | None = None
        self._years: np.ndarray | None = None
        self._log_mx_mean: float = 0.0
        self._log_mx_std: float = 1.0

    def fit(
        self,
        log_mx: np.ndarray,
        ages: np.ndarray,
        years: np.ndarray,
        exposures: np.ndarray | None = None,
        deaths: np.ndarray | None = None,
    ) -> None:
        self._ages = ages
        self._years = years
        self._n_ages = len(ages)
        self._n_years = len(years)

        self._log_mx_mean = log_mx.mean()
        self._log_mx_std = log_mx.std()
        target = (log_mx - self._log_mx_mean) / self._log_mx_std

        age_grid, year_grid = np.meshgrid(
            np.arange(self._n_ages), np.arange(self._n_years), indexing="ij"
        )
        age_idx = torch.tensor(age_grid.ravel(), dtype=torch.long)
        year_idx = torch.tensor(year_grid.ravel(), dtype=torch.long)
        y_all = torch.tensor(target.ravel(), dtype=torch.float32)

        n = len(y_all)
        split = max(1, int(0.85 * n))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed))
        train_idx, val_idx = perm[:split], perm[split:]

        torch.manual_seed(self.seed)
        self.net = _FFNNNet(
            self._n_ages, self._n_years + 50,
            self.embedding_dim, self.hidden_sizes, self.dropout,
        )
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        best_val = float("inf")
        best_state = None
        wait = 0

        self.net.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self.net(age_idx[train_idx], year_idx[train_idx])
            loss = nn.functional.mse_loss(pred, y_all[train_idx])
            loss.backward()
            optimizer.step()

            self.net.eval()
            with torch.no_grad():
                val_pred = self.net(age_idx[val_idx], year_idx[val_idx])
                val_loss = nn.functional.mse_loss(val_pred, y_all[val_idx]).item()
            self.net.train()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state:
            self.net.load_state_dict(best_state)
        self.net.eval()

    def forecast(self, h: int) -> np.ndarray:
        age_idx = torch.arange(self._n_ages, dtype=torch.long)
        results = np.zeros((self._n_ages, h))

        with torch.no_grad():
            for t in range(h):
                year_idx = torch.full((self._n_ages,), self._n_years + t, dtype=torch.long)
                pred = self.net(age_idx, year_idx).numpy()
                results[:, t] = pred * self._log_mx_std + self._log_mx_mean

        return results

    def simulate(self, h: int, n_paths: int = 1000, seed: int = 42) -> np.ndarray:
        fc = self.forecast(h)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 0.05, size=(n_paths, *fc.shape))
        return fc[None, :, :] + noise
