"""
src/models/xgboost_model.py
-----------------------------
XGBoost multi-factor oil price forecaster.

Two tasks:
  1. Regression:     predict next-N-day return (continuous)
  2. Classification: predict price direction (UP / DOWN)

Key advantages over ARIMA:
  - Handles non-linear relationships between 100+ features
  - Incorporates macro, sentiment, inventory, and technical features
  - SHAP explainability: understand WHICH factors drove each forecast
  - Robust to outliers and regime changes with tree-based learning
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, roc_auc_score, classification_report,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path("models")


class XGBoostOilForecaster:
    """
    XGBoost forecaster for oil price returns and direction.

    Predicts:
      - target_1d_return  (next-day return)
      - target_5d_return  (1-week return)
      - target_direction_1d (direction classification)
    """

    def __init__(
        self,
        reg_params: Optional[dict] = None,
        clf_params: Optional[dict] = None,
    ):
        default_reg = {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 10,
            "gamma": 0.1,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }
        default_clf = {
            **default_reg,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "scale_pos_weight": 1.0,
        }
        self.reg_params  = {**default_reg,  **(reg_params  or {})}
        self.clf_params  = {**default_clf,  **(clf_params  or {})}

        self.regressor   = None
        self.classifier  = None
        self.scaler      = StandardScaler()
        self.feature_cols = None
        self.explainer    = None

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> "XGBoostOilForecaster":
        self.feature_cols = feature_cols

        X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_val   = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        y_reg_train   = train_df["target_1d_return"].fillna(0)
        y_reg_val     = val_df["target_1d_return"].fillna(0)
        y_clf_train   = train_df["target_direction_1d"].fillna(0)
        y_clf_val     = val_df["target_direction_1d"].fillna(0)

        # ── Regressor ────────────────────────────────────────────
        print("  [1/2] Training XGBoost regressor (1-day return)...")
        self.regressor = xgb.XGBRegressor(**{
            **self.reg_params, "early_stopping_rounds": 30
        })
        self.regressor.fit(
            X_train, y_reg_train,
            eval_set=[(X_val, y_reg_val)],
            verbose=False,
        )

        # ── Classifier ───────────────────────────────────────────
        n_pos = y_clf_train.sum()
        n_neg = len(y_clf_train) - n_pos
        self.clf_params["scale_pos_weight"] = n_neg / max(n_pos, 1)

        print("  [2/2] Training XGBoost classifier (direction)...")
        self.classifier = xgb.XGBClassifier(**{
            **self.clf_params, "early_stopping_rounds": 30
        })
        self.classifier.fit(
            X_train, y_clf_train,
            eval_set=[(X_val, y_clf_val)],
            verbose=False,
        )

        # Build SHAP explainer on regressor
        self.explainer = shap.TreeExplainer(self.regressor)
        print("  ✓ XGBoost training complete")
        return self

    def predict_return(self, X: pd.DataFrame) -> np.ndarray:
        """Predict next-day return (continuous)."""
        X_clean = X[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.regressor.predict(X_clean)

    def predict_price(
        self, X: pd.DataFrame, current_prices: np.ndarray
    ) -> np.ndarray:
        """Predict next-day absolute price."""
        returns = self.predict_return(X)
        return current_prices * (1 + returns)

    def predict_direction_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of price going UP (class 1)."""
        X_clean = X[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.classifier.predict_proba(X_clean)[:, 1]

    def explain(
        self, X: pd.DataFrame, max_samples: int = 200
    ) -> pd.DataFrame:
        """Compute SHAP values for return predictions."""
        sample = X[self.feature_cols].head(max_samples).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)
        shap_values = self.explainer.shap_values(sample)
        return pd.DataFrame(shap_values, columns=self.feature_cols)

    def get_top_drivers(
        self, row: pd.Series, n: int = 5
    ) -> List[Dict]:
        """Return top N feature drivers for a single prediction."""
        X_row = pd.DataFrame(
            [row[self.feature_cols].replace([np.inf, -np.inf], 0).fillna(0)]
        )
        sv = self.explainer.shap_values(X_row)[0]
        feat_imp = pd.Series(sv, index=self.feature_cols)
        top = feat_imp.abs().nlargest(n)
        return [
            {
                "factor": feat,
                "shap_value": round(float(feat_imp[feat]), 6),
                "impact": "bullish" if feat_imp[feat] > 0 else "bearish",
                "magnitude": round(float(abs(feat_imp[feat])), 6),
            }
            for feat in top.index
        ]

    def evaluate(
        self, test_df: pd.DataFrame, current_prices: Optional[pd.Series] = None
    ) -> Dict:
        """Comprehensive evaluation on test set."""
        X_test = test_df[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Return regression metrics
        y_ret_true = test_df["target_1d_return"].fillna(0)
        y_ret_pred = self.regressor.predict(X_test)

        mae  = float(mean_absolute_error(y_ret_true, y_ret_pred))
        rmse = float(np.sqrt(mean_squared_error(y_ret_true, y_ret_pred)))

        # Price MAE (if current prices available)
        price_mae = None
        if current_prices is not None:
            prices_arr = current_prices.values
            pred_prices = prices_arr * (1 + y_ret_pred)
            true_prices = prices_arr * (1 + y_ret_true.values)
            price_mae = float(mean_absolute_error(true_prices, pred_prices))
            price_rmse = float(np.sqrt(mean_squared_error(true_prices, pred_prices)))

        # Direction classification metrics
        y_dir_true  = test_df["target_direction_1d"].fillna(0)
        y_dir_proba = self.classifier.predict_proba(X_test)[:, 1]
        y_dir_pred  = (y_dir_proba >= 0.5).astype(int)

        dir_acc = float(accuracy_score(y_dir_true, y_dir_pred))
        roc_auc = float(roc_auc_score(y_dir_true, y_dir_proba))

        result = {
            "model":       "XGBoost",
            "return_mae":  round(mae, 6),
            "return_rmse": round(rmse, 6),
            "dir_accuracy": round(dir_acc, 4),
            "roc_auc":     round(roc_auc, 4),
        }
        if price_mae is not None:
            result["price_mae_usd"]  = round(price_mae, 4)
            result["price_rmse_usd"] = round(price_rmse, 4)

        return result

    def save(self, path: Path) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "XGBoostOilForecaster":
        return joblib.load(path)
