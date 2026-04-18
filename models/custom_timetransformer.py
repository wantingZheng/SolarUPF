import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import torch
import numpy as np
from typing import Tuple
import random
import math


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
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


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Input shape:
        [B, T, d_model]

    Output shape:
        [B, T, d_model]
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerMultiQuantileNet(nn.Module):
    """
    Transformer-based multi-quantile forecasting network.

    Inputs:
        x_past   : [B, past_len, past_input_dim]
        x_future : [B, horizon, future_input_dim]

    Output:
        out      : [B, horizon, num_quantiles]
    """
    def __init__(
        self,
        past_input_dim: int,
        future_input_dim: int,
        horizon: int,
        quantiles: np.ndarray,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        pool_mode: str = "last"
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.horizon = horizon
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.hidden_dim = hidden_dim
        self.pool_mode = pool_mode

        self.input_proj = nn.Linear(past_input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, max_len=512)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
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
        # Encode the historical sequence
        x = self.input_proj(x_past)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # Pool the sequence representation
        if self.pool_mode == "mean":
            past_repr = x.mean(dim=1)
        else:
            past_repr = x[:, -1, :]

        # Encode known future features
        future_flat = x_future.reshape(x_future.size(0), -1)
        future_emb = self.future_proj(future_flat)

        # Fuse past and future representations
        fusion = torch.cat([past_repr, future_emb], dim=1)

        # Project to horizon x quantiles
        out = self.head(fusion)
        out = out.reshape(-1, self.horizon, self.num_quantiles)

        return out


class CustomTransformerTIME1:
    """
    Transformer-based probabilistic time series forecasting wrapper.

    This class keeps the same train/predict interface as the LSTM version.
    """
    def __init__(
        self,
        zhixin: np.ndarray,
        past_input_dim: int,
        future_input_dim: int,
        horizon: int = 96,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        lr: float = 1e-3,
        batch_size: int = 128,
        num_epochs: int = 40,
        dropout: float = 0.1,
        patience: int = 8,
        pool_mode: str = "last"
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
        self.lower_indices = [
            int(np.where(np.isclose(self.quantiles, (1 - c) / 2))[0][0])
            for c in self.conf_levels
        ]
        self.upper_indices = [
            int(np.where(np.isclose(self.quantiles, 1 - (1 - c) / 2))[0][0])
            for c in self.conf_levels
        ]

        self.model = TransformerMultiQuantileNet(
            past_input_dim=past_input_dim,
            future_input_dim=future_input_dim,
            horizon=horizon,
            quantiles=self.quantiles,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            pool_mode=pool_mode
        ).to(device)

    def _pinball_loss(self, pred_q: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        pred_q shape:
            [B, horizon, n_q]

        y_true shape:
            [B, horizon]
        """
        q = torch.tensor(
            self.quantiles,
            dtype=pred_q.dtype,
            device=pred_q.device
        ).view(1, 1, -1)

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

        train_loader = TorchDataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        valid_loader = TorchDataLoader(
            valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

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
            train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else np.inf

            print(
                f"Epoch [{epoch + 1:03d}/{self.num_epochs:03d}] "
                f"Train Loss: {train_loss:.6f} | Valid Loss: {val_loss:.6f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                bad_count = 0
            else:
                bad_count += 1

            if bad_count >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
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

        dummy_y = np.zeros((x_past.shape[0], self.horizon), dtype=np.float32)
        ds = SeqForecastDataset(x_past, x_future, dummy_y)
        loader = TorchDataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

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

        return (
            y_middle_matrix.astype(np.float32),
            upper_matrix.astype(np.float32),
            lower_matrix.astype(np.float32))