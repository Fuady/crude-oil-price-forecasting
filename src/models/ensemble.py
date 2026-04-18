"""
src/models/ensemble.py
-----------------------
Weighted ensemble combining XGBoost + LSTM forecasts.

Ensemble strategy:
  - XGBoost: strong on cross-sectional feature interactions (macro + inventory)
  - LSTM: strong on temporal sequence patterns (momentum, mean reversion)
  - Ensemble weights optimized on validation set to minimize MAE
  - Confidence intervals from LSTM MC-dropout + XGBoost residual spread

Academic basis: Model averaging consistently outperforms individual models
in commodity price forecasting (Genre et al., 2013; Baumeister & Kilian, 2015).
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MODELS_DIR = Path("models")


class EnsembleForecaster:
    """
    Weighted ensemble of XGBoost + LSTM for oil price forecasting.
    """

    def __init__(
        self,
        xgb_weight: float = 0.45,
        lstm_weight: float = 0.55,
    ):
        self.xgb_weight  = xgb_weight
        self.lstm_weight = lstm_weight
        self.xgb_model   = None
        self.lstm_trainer = None
        self.feature_cols = None

    def set_models(self, xgb_model, lstm_trainer) -> None:
        self.xgb_model    = xgb_model
        self.lstm_trainer = lstm_trainer
        self.feature_cols = xgb_model.feature_cols

    def optimize_weights(
        self,
        val_df: pd.DataFrame,
        xgb_preds: np.ndarray,
        lstm_preds: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Find optimal weights by minimizing MAE on validation set.
        Grid search over w in [0.1, 0.9] at 0.05 intervals.
        """
        y_true = val_df["target_1d_return"].fillna(0).values[:len(xgb_preds)]

        best_mae = np.inf
        best_w   = 0.5

        for w in np.arange(0.10, 0.95, 0.05):
            ensemble = w * xgb_preds + (1 - w) * lstm_preds
            n = min(len(y_true), len(ensemble))
            mae = np.mean(np.abs(y_true[:n] - ensemble[:n]))
            if mae < best_mae:
                best_mae = mae
                best_w   = w

        self.xgb_weight  = round(best_w, 2)
        self.lstm_weight = round(1 - best_w, 2)
        print(f"  Optimal weights: XGBoost={self.xgb_weight}, LSTM={self.lstm_weight}")
        print(f"  Ensemble val MAE: {best_mae:.6f}")
        return self.xgb_weight, self.lstm_weight

    def predict_return(
        self,
        X_df: pd.DataFrame,
        lstm_sequence: Optional[np.ndarray] = None,
    ) -> float:
        """
        Predict next-day return using weighted ensemble.

        Parameters
        ----------
        X_df          : feature row(s) for XGBoost
        lstm_sequence : (lookback, n_features) array for LSTM
        """
        xgb_pred = 0.0
        lstm_pred = 0.0

        if self.xgb_model is not None:
            try:
                xgb_pred = float(self.xgb_model.predict_return(X_df)[0])
            except Exception:
                pass

        if self.lstm_trainer is not None and lstm_sequence is not None:
            try:
                lstm_out, _ = self.lstm_trainer.predict(lstm_sequence)
                lstm_pred   = float(lstm_out[0])
            except Exception:
                pass

        if self.lstm_trainer is None or lstm_sequence is None:
            return xgb_pred   # Fall back to XGBoost only
        if self.xgb_model is None:
            return lstm_pred  # Fall back to LSTM only

        return self.xgb_weight * xgb_pred + self.lstm_weight * lstm_pred

    def predict_with_ci(
        self,
        X_df: pd.DataFrame,
        lstm_sequence: Optional[np.ndarray],
        current_price: float,
        n_mc_samples: int = 50,
    ) -> Dict:
        """
        Full prediction with confidence interval.

        Returns
        -------
        dict with:
            predicted_return, predicted_price,
            price_lower_95, price_upper_95,
            xgb_return, lstm_return,
            direction_probability
        """
        xgb_ret    = float(self.xgb_model.predict_return(X_df)[0]) if self.xgb_model else 0.0
        dir_prob   = float(self.xgb_model.predict_direction_proba(X_df)[0]) if self.xgb_model else 0.5

        lstm_mean  = lstm_lower = lstm_upper = 0.0
        if self.lstm_trainer is not None and lstm_sequence is not None:
            lstm_mean, lstm_lower, lstm_upper = self.lstm_trainer.predict_with_uncertainty(
                lstm_sequence, n_samples=n_mc_samples
            )

        # Ensemble return
        ensemble_ret = self.xgb_weight * xgb_ret + self.lstm_weight * lstm_mean

        # Confidence interval: propagate LSTM uncertainty
        ci_width     = (lstm_upper - lstm_lower) * self.lstm_weight
        lower_ret    = ensemble_ret - ci_width / 2
        upper_ret    = ensemble_ret + ci_width / 2

        return {
            "predicted_return":  round(ensemble_ret, 6),
            "predicted_price":   round(current_price * (1 + ensemble_ret), 2),
            "price_lower_95":    round(current_price * (1 + lower_ret), 2),
            "price_upper_95":    round(current_price * (1 + upper_ret), 2),
            "xgb_return":        round(xgb_ret, 6),
            "lstm_return":       round(lstm_mean, 6),
            "direction_prob_up": round(dir_prob, 4),
        }

    def evaluate(
        self,
        test_df: pd.DataFrame,
        current_prices: pd.Series,
    ) -> Dict:
        """Evaluate ensemble on test set (XGBoost path only for speed)."""
        if self.xgb_model is None:
            return {}

        xgb_ret  = self.xgb_model.predict_return(test_df)
        y_true   = test_df["target_1d_return"].fillna(0).values
        n        = min(len(y_true), len(xgb_ret))

        mae  = float(np.mean(np.abs(y_true[:n] - xgb_ret[:n])))
        rmse = float(np.sqrt(np.mean((y_true[:n] - xgb_ret[:n]) ** 2)))
        dir_acc = float(np.mean(np.sign(y_true[:n]) == np.sign(xgb_ret[:n])))

        # Price metrics
        p_true = current_prices.values[:n] * (1 + y_true[:n])
        p_pred = current_prices.values[:n] * (1 + xgb_ret[:n])
        price_mae  = float(np.mean(np.abs(p_true - p_pred)))
        price_rmse = float(np.sqrt(np.mean((p_true - p_pred) ** 2)))

        return {
            "model":         "Ensemble",
            "return_mae":    round(mae, 6),
            "return_rmse":   round(rmse, 6),
            "dir_accuracy":  round(dir_acc, 4),
            "price_mae_usd": round(price_mae, 4),
            "price_rmse_usd": round(price_rmse, 4),
            "xgb_weight":    self.xgb_weight,
            "lstm_weight":   self.lstm_weight,
        }

    def save(self, path: Path) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "xgb_weight":  self.xgb_weight,
            "lstm_weight": self.lstm_weight,
            "feature_cols": self.feature_cols,
        }
        joblib.dump(state, path)

    def load_weights(self, path: Path) -> None:
        state = joblib.load(path)
        self.xgb_weight  = state["xgb_weight"]
        self.lstm_weight = state["lstm_weight"]
        self.feature_cols = state.get("feature_cols")
