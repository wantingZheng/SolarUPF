import os
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import sklearn.metrics

import tabm as tabm_lib
import rtdl_num_embeddings


@dataclass
class TabMConfig:
    # Task
    task_type: Literal["regression", "classification"] = "regression"
    n_classes: Optional[int] = None
    share_training_batches: bool = True  # same y shared across k members

    # Feature structure
    n_num_features: int = 0
    cat_cardinalities: Optional[List[int]] = None

    # Numerical embedding
    num_embedding_type: Literal["none", "linear_relu", "periodic", "piecewise"] = "periodic"
    piecewise_bins: int = 48
    piecewise_d_embedding: int = 16

    # Training
    lr: float = 2e-3
    weight_decay: float = 3e-4
    batch_size: int = 256
    num_epochs: int = 50
    gradient_clipping_norm: Optional[float] = 1.0

    # AMP & compile
    enable_amp: bool = False
    compile_model: bool = False

    # Evaluation
    eval_batch_size: int = 8096

    # Device
    device: Optional[str] = None
    seed: int = 0


class CustomTABM1:
    """
    TabM quantile regression wrapper.

    Model output shape: (B, k, Q)
    - k: ensemble members inside TabM
    - Q: number of quantiles
    """

    def __init__(self, zhixin: List[float], config: TabMConfig):
        self.cfg = config

        # Seed
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)

        # Save confidence list
        self.zhixin = list(zhixin)

        # Build quantile list in the SAME order as model outputs.
        # Here we use: [median] + upper + lower
        # You can change this order if you want, but keep it consistent everywhere.
        upper_q = 1 - (1 - np.array(self.zhixin)) / 2  # e.g. 0.5,0.65,0.75,0.85,0.95 ...
        # lower corresponding (excluding median)
        lower_q = 1 - upper_q[1:]                      # e.g. 0.35,0.25,0.15,0.05 ...
        q_list = np.concatenate([upper_q[:1], upper_q[1:], lower_q], axis=0)  # [0.5, uppers..., lowers...]

        self.q_list = q_list.astype(np.float32)  # (Q,)
        self.Q = len(self.q_list)

        # Device
        if self.cfg.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)

        # AMP
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None

        self.amp_enabled = bool(self.cfg.enable_amp and (self.amp_dtype is not None))
        self.grad_scaler = torch.cuda.amp.GradScaler() if (self.amp_enabled and self.amp_dtype is torch.float16) else None

        # Evaluation context
        self._evaluation_mode = torch.inference_mode

        print(f"Device:        {self.device}")
        print(f"AMP:           {self.amp_enabled}{f' ({self.amp_dtype})' if self.amp_enabled else ''}")
        print(f"torch.compile: {self.cfg.compile_model}")

        self.data: Dict[str, Dict[str, torch.Tensor]] = {}
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.num_embeddings = None

    # ---------------------------
    # Public APIs
    # ---------------------------

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        patience: int = 10,
        max_epochs: int = 10_000,
        median_weight: float = 1.0,
        crossing_weight: float = 1.0,
    ) -> None:
        """
        Train with early stopping.
        y is assumed normalized (same as your pipeline).
        """
        data_numpy = {
            "train": {"x_num": x.astype(np.float32), "y": y.astype(np.float32)},
            "val": {"x_num": x_val.astype(np.float32), "y": y_val.astype(np.float32)},
        }
        self.data = self._to_tensors(data_numpy)
        self.num_embeddings = self._build_num_embeddings()

        # Build model: d_out = Q quantiles
        self.model = tabm_lib.TabM.make(
            n_num_features=self.cfg.n_num_features,
            cat_cardinalities=self.cfg.cat_cardinalities,
            d_out=self.Q,
            num_embeddings=self.num_embeddings,
        ).to(self.device)

        if self.cfg.compile_model:
            self.model = torch.compile(self.model)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        train_y = self.data["train"]["y"]
        train_size = len(train_y)
        bs = self.cfg.batch_size

        best = {"val": -math.inf}
        best_state = deepcopy(self.model.state_dict())
        remaining_patience = patience

        q_tensor = torch.as_tensor(self.q_list, device=self.device)  # (Q,)

        for epoch in range(max_epochs):
            # Batch index generation
            if self.cfg.share_training_batches:
                batches = torch.randperm(train_size, device=self.device).split(bs)
            else:
                # Different permutation for each member k
                k = self.model.backbone.k
                batches = (
                    torch.rand((train_size, k), device=self.device)
                    .argsort(dim=0)
                    .split(bs, dim=0)
                )

            # ---- Train one epoch ----
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for idx in batches:
                self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(self.device.type, enabled=self.amp_enabled, dtype=self.amp_dtype):
                    pred = self._apply_model(self.data["train"]["x_num"], self.data["train"].get("x_cat", None), idx)
                    # pred: (B, k, Q)
                    loss = self._quantile_loss(
                        pred=pred,
                        y_true=train_y[idx],
                        quantiles=q_tensor,
                        median_weight=median_weight,
                        crossing_weight=crossing_weight,
                    )

                if self.grad_scaler is None:
                    loss.backward()
                    if self.cfg.gradient_clipping_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clipping_norm)
                    self.optimizer.step()
                else:
                    self.grad_scaler.scale(loss).backward()
                    if self.cfg.gradient_clipping_norm is not None:
                        self.grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clipping_norm)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()

                total_loss += float(loss.item())
                n_batches += 1

            train_loss = total_loss / max(1, n_batches)

            # ---- Validate ----
            val_score = self._evaluate_rmse_plus_quantile("val", q_tensor, median_weight, crossing_weight)
            improved = val_score > best["val"]

            print(
                f'{"*" if improved else " "}'
                f" epoch={epoch:04d}"
                f" train_loss={train_loss:.5f}"
                f" val_score={val_score:.5f}"
                f" (best={best['val']:.5f})"
            )

            if improved:
                best["val"] = val_score
                best_state = deepcopy(self.model.state_dict())
                remaining_patience = patience
            else:
                remaining_patience -= 1
                if remaining_patience < 0:
                    print(f">>> Early stopping at epoch {epoch}")
                    break

        self.model.load_state_dict(best_state)
        print(f"[Done] best_val_score={best['val']:.6f}")

    @torch.no_grad()
    def predict(self, X: np.ndarray, y_mean: float, y_std: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return:
          y_median: (N,)
          upper:    (N, n_upper)
          lower:    (N, n_upper)  # symmetric count
        """
        assert self.model is not None

        self.model.eval()
        X_t = torch.as_tensor(X, device=self.device, dtype=torch.float32)

        with self._evaluation_mode():
            out = self.model(X_t)  # (N, k, Q)

        out = out.mean(dim=1)  # (N, Q)
        out = out.detach().cpu().numpy()

        # De-normalize
        out = out * y_std + y_mean

        # Output order: [median] + uppers + lowers
        n_upper = len(self.zhixin) - 1
        y_median = out[:, 0]                 # (N,)
        upper = out[:, 1 : 1 + n_upper]      # (N, n_upper)
        lower = out[:, 1 + n_upper :]        # (N, n_upper)

        return y_median, upper, lower

    def save_model(self, path: str) -> None:
        assert self.model is not None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "q_list": self.q_list, "cfg": self.cfg.__dict__}, path)
        print(f"[OK] Saved to: {path}")

    def load_model(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_list = ckpt["q_list"]
        self.Q = len(self.q_list)
        # Rebuild model skeleton then load
        self.model = tabm_lib.TabM.make(
            n_num_features=self.cfg.n_num_features,
            cat_cardinalities=self.cfg.cat_cardinalities,
            d_out=self.Q,
            num_embeddings=self.num_embeddings,
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        print(f"[OK] Loaded from: {path}")

    # ---------------------------
    # Internals
    # ---------------------------

    def _to_tensors(self, data_numpy: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, torch.Tensor]]:
        data: Dict[str, Dict[str, torch.Tensor]] = {}
        for part, d in data_numpy.items():
            data[part] = {k: torch.as_tensor(v, device=self.device) for k, v in d.items()}
            if "y" in data[part]:
                data[part]["y"] = data[part]["y"].float()
        return data

    def _build_num_embeddings(self):
        if self.cfg.num_embedding_type == "none":
            return None
        if self.cfg.num_embedding_type == "linear_relu":
            return rtdl_num_embeddings.LinearReLUEmbeddings(self.cfg.n_num_features)
        if self.cfg.num_embedding_type == "periodic":
            return rtdl_num_embeddings.PeriodicEmbeddings(self.cfg.n_num_features, lite=False)
        if self.cfg.num_embedding_type == "piecewise":
            bins = rtdl_num_embeddings.compute_bins(self.data["train"]["x_num"], n_bins=self.cfg.piecewise_bins)
            return rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                bins, d_embedding=self.cfg.piecewise_d_embedding, activation=False, version="B"
            )
        raise ValueError(f"Unknown num_embedding_type: {self.cfg.num_embedding_type}")

    def _apply_model(self, xnum: torch.Tensor, xcat: Optional[torch.Tensor], idx: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        out = self.model(xnum[idx], xcat[idx] if xcat is not None else None)  # (B, k, Q)
        return out.float()

    def _quantile_loss(
        self,
        pred: torch.Tensor,          # (B, k, Q)
        y_true: torch.Tensor,        # (B,)
        quantiles: torch.Tensor,     # (Q,)
        median_weight: float = 1.0,
        crossing_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Pinball loss + crossing penalty.
        """
        assert pred.dim() == 3, f"Expected (B,k,Q), got {pred.shape}"
        B, k, Q = pred.shape

        # Flatten (B,k,Q) -> (B*k, Q)
        pred_flat = pred.reshape(B * k, Q)

        # Expand y: (B,) -> (B*k, 1) -> (B*k, Q)
        if self.cfg.share_training_batches:
            y_flat = y_true.repeat_interleave(k).reshape(-1, 1)  # (B*k,1)
        else:
            # If share_training_batches=False, y_true should already be (B,k)
            y_flat = y_true.reshape(-1, 1)

        y_q = y_flat.expand(-1, Q)  # (B*k, Q)

        # Pinball loss
        # errors = y - pred
        errors = y_q - pred_flat
        q = quantiles.view(1, -1)  # (1,Q)
        loss = torch.maximum((q - 1.0) * errors, q * errors)  # (B*k,Q)

        # Extra weight for median (q == 0.5)
        if median_weight != 1.0:
            median_mask = (torch.abs(quantiles - 0.5) < 1e-12).float().view(1, -1)
            loss = loss * (1.0 + (median_weight - 1.0) * median_mask)

        loss_pinball = loss.mean()

        # Crossing penalty: enforce monotonicity in quantiles (sorted by q)
        sorted_idx = torch.argsort(quantiles)  # (Q,)
        pred_sorted = pred_flat.index_select(dim=1, index=sorted_idx)
        crossing = torch.relu(pred_sorted[:, :-1] - pred_sorted[:, 1:]).mean()

        return loss_pinball + crossing_weight * crossing

    @torch.no_grad()
    def _evaluate_rmse_plus_quantile(
        self,
        part: str,
        quantiles: torch.Tensor,
        median_weight: float,
        crossing_weight: float,
    ) -> float:
        """
        Score = -RMSE(median) - quantile_loss (normalized space).
        Higher is better.
        """
        assert self.model is not None
        self.model.eval()

        xnum = self.data[part]["x_num"]
        xcat = self.data[part].get("x_cat", None)
        y_true = self.data[part]["y"]  # normalized

        # Forward in batches
        outs = []
        for idx in torch.arange(len(xnum), device=self.device).split(self.cfg.eval_batch_size):
            out = self._apply_model(xnum, xcat, idx)  # (b,k,Q)
            outs.append(out)
        pred = torch.cat(outs, dim=0)  # (N,k,Q)

        # Median prediction for RMSE (normalized space)
        pred_mean = pred.mean(dim=1)         # (N,Q)
        y_pred_median = pred_mean[:, 0]      # (N,)

        rmse = torch.sqrt(torch.mean((y_true - y_pred_median) ** 2)).item()

        # Quantile loss in normalized space
        qloss = self._quantile_loss(pred=pred, y_true=y_true, quantiles=quantiles,
                                    median_weight=median_weight, crossing_weight=crossing_weight).item()
        # return float(-qloss)

        return float(-rmse - qloss)