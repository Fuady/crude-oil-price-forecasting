"""
src/models/baseline.py
-----------------------
Baseline forecasting models for oil price prediction.

Two baselines:
  1. Naïve (Last Value): forecast = last known price
     - The hardest baseline to beat in financial time series
     - Many sophisticated models fail to outperform this
  2. ARIMA: classical statistical time-series model
     - Standard benchmark in commodity price forecasting
     - Used as comparison in academic papers

These set the performance floor our ML models must exceed.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")


class NaiveForecaster:
    """
    Naïve (last-value / random-walk) forecaster.
    Forecast = last observed price for all horizons.

    This is the canonical benchmark in financial forecasting:
    if your model can't beat this, it has no predictive value.
    """

    def __init__(self):
        self.last_price = None

    def fit(self, prices: pd.Series) -> "NaiveForecaster":
        self.last_price = float(prices.iloc[-1])
        self.price_series = prices.copy()
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Return last known price repeated for each horizon step."""
        return np.full(horizon, self.last_price)

    def predict_return_direction(self, horizon: int = 1) -> np.ndarray:
        """Naïve: predict no change (direction = 0 = neutral)."""
        return np.zeros(horizon, dtype=int)

    def evaluate_walk_forward(
        self,
        prices: pd.Series,
        horizon: int = 1,
        min_train: int = 252,
    ) -> Dict:
        """
        Walk-forward evaluation on historical data.
        For each date from min_train onward, predict the next price.
        """
        actuals, preds = [], []

        for i in range(min_train, len(prices) - horizon):
            train_slice = prices.iloc[:i]
            self.fit(train_slice)
            pred = self.predict(horizon)[0]
            actual = float(prices.iloc[i + horizon - 1])
            preds.append(pred)
            actuals.append(actual)

        actuals = np.array(actuals)
        preds   = np.array(preds)

        mae  = float(np.mean(np.abs(actuals - preds)))
        rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
        dir_acc = float(np.mean(
            np.sign(np.diff(actuals)) == np.sign(np.diff(preds))
        )) if len(actuals) > 1 else 0.0

        return {
            "model":      "NaiveLastValue",
            "mae":        round(mae, 4),
            "rmse":       round(rmse, 4),
            "dir_acc":    round(dir_acc, 4),
            "n_forecasts": len(actuals),
        }


class ARIMAForecaster:
    """
    ARIMA(p,d,q) forecaster for oil price time series.

    Auto-selects order using AIC minimisation if not provided.
    Standard in academic commodity price forecasting papers.

    Limitations (why we need ML):
    - Assumes linear dynamics
    - Cannot incorporate exogenous variables (macro, sentiment)
    - Constant variance assumption violated by oil market regimes
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (2, 1, 2),
        auto_order: bool = False,
    ):
        self.order = order
        self.auto_order = auto_order
        self.model = None
        self.result = None
        self.fitted_values = None
        self.last_train_date = None

    def _auto_select_order(
        self, prices: pd.Series
    ) -> Tuple[int, int, int]:
        """
        Simple auto-order selection by AIC over a grid.
        Tries p in 0-3, d in 0-2, q in 0-3.
        """
        from statsmodels.tsa.arima.model import ARIMA

        best_aic = np.inf
        best_order = (1, 1, 1)

        for p in range(4):
            for d in range(2):
                for q in range(4):
                    try:
                        m = ARIMA(prices, order=(p, d, q)).fit(method="innovations_mle")
                        if m.aic < best_aic:
                            best_aic = m.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue

        return best_order

    def fit(self, prices: pd.Series) -> "ARIMAForecaster":
        from statsmodels.tsa.arima.model import ARIMA

        if self.auto_order:
            self.order = self._auto_select_order(prices.iloc[-252:])

        self.model = ARIMA(prices, order=self.order)
        self.result = self.model.fit(method="innovations_mle")
        self.last_train_date = prices.index[-1]
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("Call fit() first.")
        forecast = self.result.forecast(steps=horizon)
        return np.array(forecast)

    def predict_with_ci(
        self, horizon: int = 1, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (point forecast, lower CI, upper CI)."""
        fc = self.result.get_forecast(steps=horizon)
        mean = np.array(fc.predicted_mean)
        ci   = fc.conf_int(alpha=alpha)
        return mean, np.array(ci.iloc[:, 0]), np.array(ci.iloc[:, 1])

    def evaluate_walk_forward(
        self,
        prices: pd.Series,
        horizon: int = 1,
        min_train: int = 252,
        step: int = 5,   # refit every 5 days for speed
    ) -> Dict:
        """Walk-forward evaluation."""
        actuals, preds = [], []

        print(f"  ARIMA walk-forward ({len(prices) - min_train} steps)...")
        for i in range(min_train, len(prices) - horizon, step):
            train_slice = prices.iloc[:i]
            try:
                self.fit(train_slice)
                pred = self.predict(horizon)[0]
            except Exception:
                pred = float(train_slice.iloc[-1])   # Fallback to naïve
            actual = float(prices.iloc[i + horizon - 1])
            preds.append(pred)
            actuals.append(actual)

        actuals = np.array(actuals)
        preds   = np.array(preds)

        mae  = float(np.mean(np.abs(actuals - preds)))
        rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
        dir_acc = float(np.mean(
            np.sign(np.diff(actuals)) == np.sign(np.diff(preds))
        )) if len(actuals) > 1 else 0.0

        return {
            "model":      f"ARIMA{self.order}",
            "mae":        round(mae, 4),
            "rmse":       round(rmse, 4),
            "dir_acc":    round(dir_acc, 4),
            "n_forecasts": len(actuals),
        }
