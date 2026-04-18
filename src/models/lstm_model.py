"""
src/models/lstm_model.py
-------------------------
LSTM (Long Short-Term Memory) model for oil price sequence forecasting.

Why LSTM for commodity prices?
  - Captures long-range temporal dependencies (momentum, mean reversion)
  - Handles multiple input features simultaneously (technical + macro + sentiment)
  - Can learn non-linear regime-dependent patterns (contango, backwardation)
  - Standard approach in recent O&G price forecasting literature

Architecture:
  Input: (batch, lookback=60, n_features)
    → LSTM layer 1 (128 hidden, dropout=0.3)
    → LSTM layer 2 (64 hidden, dropout=0.2)
    → Attention layer (learns which time steps matter most)
    → Fully connected: 32 → 16 → output
  Output: (batch, forecast_horizon) — multi-step price return forecast

Attention mechanism: inspired by Attention-based LSTM for oil price
forecasting (Wang et al., 2022 - Applied Energy).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import joblib

MODELS_DIR = Path("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OilPriceDataset(Dataset):
    """
    Sliding window dataset for LSTM training.
    Each item: (features_window, target_returns)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_1d_return",
        lookback: int = 60,
        horizon: int = 1,
    ):
        self.lookback = lookback
        self.horizon  = horizon

        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
        y = df[target_col].fillna(0).values.astype(np.float32)

        # Normalize features (per-feature z-score using training stats)
        self.mean = X.mean(axis=0)
        self.std  = np.where(X.std(axis=0) > 1e-8, X.std(axis=0), 1.0)
        X_norm = (X - self.mean) / self.std

        self.sequences = []
        self.targets   = []

        for i in range(lookback, len(X_norm) - horizon + 1):
            self.sequences.append(X_norm[i - lookback : i])
            self.targets.append(y[i : i + horizon])

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets   = np.array(self.targets,   dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx]),
        )


# ---------------------------------------------------------------------------
# Attention mechanism
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Soft attention over the LSTM hidden state sequence.
    Learns which historical time steps are most informative for forecasting.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden)
        attn_weights = self.attention(lstm_output)          # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)   # normalize over time
        context = (attn_weights * lstm_output).sum(dim=1)   # (batch, hidden)
        return context, attn_weights.squeeze(-1)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class LSTMOilForecaster(nn.Module):
    """
    Two-layer LSTM with temporal attention for oil price forecasting.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        hidden_size_2: int = 64,
        output_size: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm1 = nn.LSTM(
            n_features, hidden_size,
            batch_first=True, dropout=0.0
        )
        self.lstm2 = nn.LSTM(
            hidden_size, hidden_size_2,
            batch_first=True, dropout=0.0
        )
        self.attention = TemporalAttention(hidden_size_2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.7)

        self.head = nn.Sequential(
            nn.Linear(hidden_size_2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)

        out2, _ = self.lstm2(out1)
        out2 = self.dropout2(out2)

        context, attn_weights = self.attention(out2)
        output = self.head(context)

        return output, attn_weights


# ---------------------------------------------------------------------------
# Trainer wrapper
# ---------------------------------------------------------------------------

class LSTMTrainer:

    def __init__(
        self,
        n_features: int,
        lookback: int = 60,
        horizon: int = 1,
        hidden_size: int = 128,
        device: torch.device = DEVICE,
    ):
        self.n_features  = n_features
        self.lookback    = lookback
        self.horizon     = horizon
        self.device      = device
        self.feature_cols = None

        self.model = LSTMOilForecaster(
            n_features=n_features,
            hidden_size=hidden_size,
            output_size=horizon,
        ).to(device)

        self.history = {"train_loss": [], "val_loss": []}
        self.norm_mean = None
        self.norm_std  = None

    def fit(
        self,
        train_ds: OilPriceDataset,
        val_ds: OilPriceDataset,
        epochs: int = 80,
        lr: float = 5e-4,
        batch_size: int = 64,
        patience: int = 15,
        verbose: bool = True,
    ) -> None:
        # Store normalization stats from training dataset
        self.norm_mean = train_ds.mean
        self.norm_std  = train_ds.std

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=7, factor=0.5, min_lr=1e-6
        )
        criterion = nn.HuberLoss(delta=0.01)  # Huber loss: robust to oil price spikes

        best_val   = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                pred, _ = self.model(X_batch)
                loss = criterion(pred.squeeze(), y_batch.squeeze())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            avg_train = np.mean(train_losses)
            self.history["train_loss"].append(avg_train)

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    pred, _ = self.model(X_batch)
                    val_losses.append(criterion(pred.squeeze(), y_batch.squeeze()).item())
            avg_val = np.mean(val_losses)
            self.history["val_loss"].append(avg_val)
            scheduler.step(avg_val)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} | "
                      f"Train: {avg_train:.6f} | Val: {avg_val:.6f}")

            if avg_val < best_val - 1e-7:
                best_val   = avg_val
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on a (seq_len, n_features) array.
        Returns (predictions, attention_weights).
        """
        self.model.eval()
        X_norm = (X - self.norm_mean) / self.norm_std
        X_tensor = torch.FloatTensor(X_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred, attn = self.model(X_tensor)
        return pred.cpu().numpy()[0], attn.cpu().numpy()[0]

    def predict_with_uncertainty(
        self, X: np.ndarray, n_samples: int = 50
    ) -> Tuple[float, float, float]:
        """
        Monte Carlo dropout for prediction intervals.
        Run forward pass N times with dropout active.
        Returns (mean, lower_95, upper_95).
        """
        self.model.train()   # Enable dropout
        X_norm = (X - self.norm_mean) / self.norm_std
        X_tensor = torch.FloatTensor(X_norm).unsqueeze(0).to(self.device)

        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred, _ = self.model(X_tensor)
                preds.append(pred.cpu().numpy()[0, 0])

        self.model.eval()
        preds = np.array(preds)
        return float(np.mean(preds)), float(np.percentile(preds, 2.5)), float(np.percentile(preds, 97.5))

    def evaluate(self, test_ds: OilPriceDataset) -> Dict:
        """Evaluate on test dataset."""
        loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        self.model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                pred, _ = self.model(X_batch.to(self.device))
                all_preds.extend(pred.cpu().numpy()[:, 0])
                all_targets.extend(y_batch.numpy()[:, 0])

        preds   = np.array(all_preds)
        targets = np.array(all_targets)

        mae  = float(np.mean(np.abs(targets - preds)))
        rmse = float(np.sqrt(np.mean((targets - preds) ** 2)))
        dir_acc = float(np.mean(np.sign(targets) == np.sign(preds)))

        return {
            "model":       "LSTM",
            "return_mae":  round(mae, 6),
            "return_rmse": round(rmse, 6),
            "dir_accuracy": round(dir_acc, 4),
        }

    def save(self, path: Path) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state":  self.model.state_dict(),
            "n_features":   self.n_features,
            "lookback":     self.lookback,
            "horizon":      self.horizon,
            "norm_mean":    self.norm_mean,
            "norm_std":     self.norm_std,
            "feature_cols": self.feature_cols,
            "history":      self.history,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "LSTMTrainer":
        state = torch.load(path, map_location=DEVICE)
        trainer = cls(
            n_features=state["n_features"],
            lookback=state["lookback"],
            horizon=state["horizon"],
        )
        trainer.model.load_state_dict(state["model_state"])
        trainer.norm_mean    = state["norm_mean"]
        trainer.norm_std     = state["norm_std"]
        trainer.feature_cols = state["feature_cols"]
        trainer.history      = state["history"]
        trainer.model.eval()
        return trainer
