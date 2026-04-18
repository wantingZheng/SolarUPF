import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import pandas as pd
import torch
import numpy as np
from typing import Tuple
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class SeqForecastDataset(Dataset):
    def __init__(self, x_past: np.ndarray, x_future: np.ndarray, y: np.ndarray):
        self.x_past = torch.tensor(x_past, dtype=torch.float32)
        self.x_future = torch.tensor(x_future, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x_past.shape[0]

    def __getitem__(self, idx):
        return self.x_past[idx], self.x_future[idx], self.y[idx]


class LSTMMultiQuantileNet(nn.Module):
    def __init__(
        self,
        past_input_dim: int,
        future_input_dim: int,
        horizon: int,
        quantiles: np.ndarray,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        self.horizon = horizon
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.encoder = nn.LSTM(
            input_size=past_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.future_proj = nn.Sequential(
            nn.Linear(horizon * future_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, horizon * self.num_quantiles)
        )

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x_past)
        h_last = h_n[-1]

        future_flat = x_future.reshape(x_future.size(0), -1)
        future_emb = self.future_proj(future_flat)

        fusion = torch.cat([h_last, future_emb], dim=1)
        out = self.head(fusion)
        out = out.reshape(-1, self.horizon, self.num_quantiles)
        return out


class CustomLSTMTIME1:
    def __init__(
        self,
        zhixin: np.ndarray,
        past_input_dim: int,
        future_input_dim: int,
        horizon: int = 96,
        hidden_dim: int = 64,
        num_layers: int = 1,
        lr: float = 1e-3,
        batch_size: int = 128,
        num_epochs: int = 40,
        dropout: float = 0.2,
        patience: int = 8
    ):
        self.zhixin = np.asarray(zhixin, dtype=float)
        self.conf_levels = np.asarray(zhixin[1:], dtype=float)
        self.horizon = horizon
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience

        quantiles = [0.5]
        quantiles += [float((1 - c) / 2) for c in self.conf_levels]
        quantiles += [float(1 - (1 - c) / 2) for c in self.conf_levels]
        self.quantiles = np.array(sorted(set(quantiles)), dtype=np.float32)

        self.median_idx = int(np.where(np.isclose(self.quantiles, 0.5))[0][0])
        self.lower_indices = [int(np.where(np.isclose(self.quantiles, (1 - c) / 2))[0][0]) for c in self.conf_levels]
        self.upper_indices = [int(np.where(np.isclose(self.quantiles, 1 - (1 - c) / 2))[0][0]) for c in self.conf_levels]

        self.model = LSTMMultiQuantileNet(
            past_input_dim=past_input_dim,
            future_input_dim=future_input_dim,
            horizon=horizon,
            quantiles=self.quantiles,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

    def _pinball_loss(self, pred_q: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        pred_q: [B, horizon, n_q]
        y_true: [B, horizon]
        """
        q = torch.tensor(self.quantiles, dtype=pred_q.dtype, device=pred_q.device).view(1, 1, -1)
        y_true = y_true.unsqueeze(-1)
        error = y_true - pred_q
        loss = torch.maximum(q * error, (q - 1.0) * error)
        return loss.mean()

    def train(
        self,
        train_x_past: np.ndarray,
        train_x_future: np.ndarray,
        train_y: np.ndarray,
        valid_x_past: np.ndarray,
        valid_x_future: np.ndarray,
        valid_y: np.ndarray
    ) -> None:
        train_ds = SeqForecastDataset(train_x_past, train_x_future, train_y)
        valid_ds = SeqForecastDataset(valid_x_past, valid_x_future, valid_y)

        train_loader = TorchDataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        valid_loader = TorchDataLoader(valid_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val = np.inf
        best_state = None
        bad_count = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            train_losses = []

            for batch_x_past, batch_x_future, batch_y in train_loader:
                batch_x_past = batch_x_past.to(device)
                batch_x_future = batch_x_future.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                pred_q = self.model(batch_x_past, batch_x_future)
                loss = self._pinball_loss(pred_q, batch_y)
                loss.backward()
                optimizer.step()

                train_losses.append(float(loss.item()))

            self.model.eval()
            valid_losses = []
            with torch.no_grad():
                for batch_x_past, batch_x_future, batch_y in valid_loader:
                    batch_x_past = batch_x_past.to(device)
                    batch_x_future = batch_x_future.to(device)
                    batch_y = batch_y.to(device)

                    pred_q = self.model(batch_x_past, batch_x_future)
                    loss = self._pinball_loss(pred_q, batch_y)
                    valid_losses.append(float(loss.item()))

            val_loss = float(np.mean(valid_losses)) if len(valid_losses) > 0 else np.inf

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad_count = 0
            else:
                bad_count += 1

            if bad_count >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(
        self,
        x_past: np.ndarray,
        x_future: np.ndarray,
        y_mean: float,
        y_std: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        ds = SeqForecastDataset(x_past, x_future, np.zeros((x_past.shape[0], self.horizon), dtype=np.float32))
        loader = TorchDataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

        pred_q_list = []
        with torch.no_grad():
            for batch_x_past, batch_x_future, _ in loader:
                batch_x_past = batch_x_past.to(device)
                batch_x_future = batch_x_future.to(device)
                pred_q = self.model(batch_x_past, batch_x_future)
                pred_q_list.append(pred_q.detach().cpu().numpy())

        pred_q_all = np.concatenate(pred_q_list, axis=0)
        pred_q_all = pred_q_all * y_std + y_mean

        y_middle_matrix = pred_q_all[:, :, self.median_idx]
        lower_matrix = pred_q_all[:, :, self.lower_indices]
        upper_matrix = pred_q_all[:, :, self.upper_indices]

        return y_middle_matrix.astype(np.float32), upper_matrix.astype(np.float32), lower_matrix.astype(np.float32)

