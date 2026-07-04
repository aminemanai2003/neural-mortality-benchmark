"""Neural models that forecast the Lee-Carter kt index.

Replace the random walk with drift with a neural network:
LSTM, GRU, BiLSTM, or Transformer encoder.
The LC structure (ax + bx*kt) is preserved -- only kt forecasting changes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mortality.models.base import MortalityModel


class _LSTMNet(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class _GRUNet(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class _BiLSTMNet(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            1, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class _TransformerNet(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        seq_len = x.size(1)
        x = x + self.pos_enc[:, :seq_len, :]
        out = self.transformer(x)
        return self.fc(out[:, -1, :])


NET_REGISTRY = {
    "lstm": _LSTMNet,
    "gru": _GRUNet,
    "bilstm": _BiLSTMNet,
    "transformer": _TransformerNet,
}


class NeuralKtModel(MortalityModel):
    """Neural forecaster for the Lee-Carter kt index."""

    def __init__(
        self,
        arch: str = "lstm",
        hidden_size: int = 64,
        num_layers: int = 2,
        lookback: int = 20,
        lr: float = 0.001,
        epochs: int = 200,
        patience: int = 20,
        seed: int = 42,
        d_model: int = 64,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        self.arch = arch
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lookback = lookback
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout

        self.ax: np.ndarray | None = None
        self.bx: np.ndarray | None = None
        self.kt: np.ndarray | None = None
        self.kt_mean: float = 0.0
        self.kt_std: float = 1.0
        self.net: nn.Module | None = None
        self.name = f"{arch}_kt"

    def _build_net(self) -> nn.Module:
        if self.arch == "transformer":
            return _TransformerNet(self.d_model, self.nhead, self.num_layers, self.dropout)
        return NET_REGISTRY[self.arch](self.hidden_size, self.num_layers, self.dropout)

    def _make_sequences(self, kt_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(kt_norm) - self.lookback):
            X.append(kt_norm[i : i + self.lookback])
            y.append(kt_norm[i + self.lookback])
        return np.array(X), np.array(y)

    def fit(
        self,
        log_mx: np.ndarray,
        ages: np.ndarray,
        years: np.ndarray,
        exposures: np.ndarray | None = None,
        deaths: np.ndarray | None = None,
    ) -> None:
        self.ax = log_mx.mean(axis=1)
        centered = log_mx - self.ax[:, None]
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        bx_raw = U[:, 0]
        kt_raw = S[0] * Vt[0, :]
        self.bx = bx_raw / bx_raw.sum()
        self.kt = kt_raw * bx_raw.sum()

        self.kt_mean = self.kt.mean()
        self.kt_std = self.kt.std()
        kt_norm = (self.kt - self.kt_mean) / self.kt_std

        X, y = self._make_sequences(kt_norm)
        if len(X) < 5:
            self.net = None
            return

        torch.manual_seed(self.seed)
        X_t = torch.tensor(X[:, :, None], dtype=torch.float32)
        y_t = torch.tensor(y[:, None], dtype=torch.float32)

        split = max(1, int(0.8 * len(X_t)))
        X_train, y_train = X_t[:split], y_t[:split]
        X_val, y_val = X_t[split:], y_t[split:]

        self.net = self._build_net()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        best_val_loss = float("inf")
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
                    val_pred = self.net(X_val)
                    val_loss = nn.functional.mse_loss(val_pred, y_val).item()
                self.net.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()

    def _forecast_kt(self, h: int) -> np.ndarray:
        if self.net is None:
            dkt = np.diff(self.kt)
            drift = dkt.mean()
            return self.kt[-1] + drift * np.arange(1, h + 1)

        kt_norm = (self.kt - self.kt_mean) / self.kt_std
        context = list(kt_norm[-self.lookback:])
        forecasts = []

        with torch.no_grad():
            for _ in range(h):
                x = torch.tensor(
                    np.array(context[-self.lookback:])[None, :, None],
                    dtype=torch.float32,
                )
                pred = self.net(x).item()
                forecasts.append(pred)
                context.append(pred)

        return np.array(forecasts) * self.kt_std + self.kt_mean

    def forecast(self, h: int) -> np.ndarray:
        kt_fc = self._forecast_kt(h)
        return self.ax[:, None] + self.bx[:, None] * kt_fc[None, :]

    def simulate(self, h: int, n_paths: int = 1000, seed: int = 42) -> np.ndarray:
        kt_fc = self._forecast_kt(h)
        dkt = np.diff(self.kt)
        sigma = dkt.std(ddof=1)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, sigma, size=(n_paths, h))
        kt_paths = kt_fc[None, :] + np.cumsum(noise, axis=1)
        return self.ax[None, :, None] + self.bx[None, :, None] * kt_paths[:, None, :]
