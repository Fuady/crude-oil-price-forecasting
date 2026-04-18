"""
tests/test_models.py
---------------------
Unit tests for forecasting models, signal generator, and backtester.
Run: pytest tests/test_models.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baseline import NaiveForecaster, ARIMAForecaster
from src.signals.signal_generator import generate_signal, Signal


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def price_series():
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = 80 + np.cumsum(np.random.normal(0, 1.0, n))
    return pd.Series(np.clip(prices, 30, 150), index=dates, name="close")


@pytest.fixture
def feature_row():
    """A realistic feature row for signal generation testing."""
    return pd.Series({
        "close":               82.50,
        "rsi_14":              48.0,
        "macd_hist":           0.25,
        "ema_cross_50_200":    1,
        "bb_pct_b":            0.45,
        "sentiment_score":     0.15,
        "sentiment_ma5":       0.12,
        "eia_weekly_change_mb": -2.5,
        "eia_large_draw":      0,
        "eia_large_build":     0,
        "eia_above_seasonal":  0,
        "dxy_above_ma":        0,
        "yield_inverted":      0,
        "atr_pct":             0.015,
        "high_vol_regime":     0,
        "roc_5d":              0.02,
        "extreme_bearish_sentiment": 0,
        "extreme_bullish_sentiment": 0,
    })


@pytest.fixture
def mini_feature_df():
    """A small DataFrame suitable for XGBoost training."""
    from src.features.technical_indicators import (
        add_moving_averages, add_rsi, add_macd, add_bollinger_bands,
        add_atr, add_rate_of_change, add_historical_volatility,
        add_volume_indicators, add_support_resistance,
    )
    from src.features.macro_features import add_calendar_features, add_lagged_features

    np.random.seed(42)
    n = 400
    dates  = pd.bdate_range("2021-01-01", periods=n)
    close  = 80 + np.cumsum(np.random.normal(0, 1.2, n))
    close  = np.clip(close, 30, 150)
    df = pd.DataFrame({
        "open": close * (1 + np.random.normal(0, 0.003, n)),
        "high": close * (1 + np.random.uniform(0.001, 0.01, n)),
        "low":  close * (1 - np.random.uniform(0.001, 0.01, n)),
        "close": close,
        "volume": np.random.randint(100000, 500000, n),
    }, index=dates)

    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_rate_of_change(df)
    df = add_historical_volatility(df)
    df = add_volume_indicators(df)
    df = add_support_resistance(df)
    df = add_calendar_features(df)
    df = add_lagged_features(df)

    df = df.dropna(subset=["sma_200", "target_1d_return"])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Naive Forecaster
# ─────────────────────────────────────────────────────────────────────────────

class TestNaiveForecaster:

    def test_fit_stores_last_price(self, price_series):
        model = NaiveForecaster()
        model.fit(price_series)
        assert model.last_price == pytest.approx(float(price_series.iloc[-1]), rel=1e-6)

    def test_predict_returns_constant(self, price_series):
        model = NaiveForecaster()
        model.fit(price_series)
        preds = model.predict(horizon=5)
        assert len(preds) == 5
        assert np.all(preds == preds[0])

    def test_predict_equals_last_price(self, price_series):
        model = NaiveForecaster()
        model.fit(price_series)
        preds = model.predict(7)
        assert preds[0] == pytest.approx(float(price_series.iloc[-1]), rel=1e-6)

    def test_walk_forward_returns_dict(self, price_series):
        model = NaiveForecaster()
        result = model.evaluate_walk_forward(price_series, horizon=1, min_train=100)
        assert "mae" in result
        assert "rmse" in result
        assert "dir_acc" in result
        assert result["mae"] >= 0
        assert result["rmse"] >= result["mae"]


# ─────────────────────────────────────────────────────────────────────────────
# ARIMA Forecaster
# ─────────────────────────────────────────────────────────────────────────────

class TestARIMAForecaster:

    def test_fit_and_predict(self, price_series):
        model = ARIMAForecaster(order=(1, 1, 1))
        model.fit(price_series.iloc[:100])
        preds = model.predict(horizon=5)
        assert len(preds) == 5
        assert np.isfinite(preds).all()

    def test_predict_positive_prices(self, price_series):
        model = ARIMAForecaster(order=(2, 1, 1))
        model.fit(price_series.iloc[:150])
        preds = model.predict(3)
        assert (preds > 0).all()

    def test_predict_with_ci_shape(self, price_series):
        model = ARIMAForecaster(order=(1, 1, 1))
        model.fit(price_series.iloc[:100])
        mean, lower, upper = model.predict_with_ci(horizon=3)
        assert len(mean) == 3
        assert (upper >= lower).all()


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost Forecaster
# ─────────────────────────────────────────────────────────────────────────────

class TestXGBoostForecaster:

    def test_fit_and_predict_return(self, mini_feature_df):
        from src.models.xgboost_model import XGBoostOilForecaster
        df = mini_feature_df
        feat_cols = [c for c in df.columns
                     if not c.startswith("target_") and c != "close"
                     and c not in ("open", "high", "low", "volume")]

        n = len(df)
        train = df.iloc[:int(n * 0.7)]
        val   = df.iloc[int(n * 0.7):int(n * 0.85)]
        test  = df.iloc[int(n * 0.85):]

        model = XGBoostOilForecaster(reg_params={"n_estimators": 30})
        model.fit(train, val, feat_cols)

        preds = model.predict_return(test)
        assert len(preds) == len(test)
        assert np.isfinite(preds).all()

    def test_direction_prob_range(self, mini_feature_df):
        from src.models.xgboost_model import XGBoostOilForecaster
        df = mini_feature_df
        feat_cols = [c for c in df.columns
                     if not c.startswith("target_") and c != "close"
                     and c not in ("open", "high", "low", "volume")]

        n = len(df)
        train = df.iloc[:int(n * 0.7)]
        val   = df.iloc[int(n * 0.7):int(n * 0.85)]
        test  = df.iloc[int(n * 0.85):]

        model = XGBoostOilForecaster(reg_params={"n_estimators": 30})
        model.fit(train, val, feat_cols)
        probs = model.predict_direction_proba(test)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_evaluate_returns_metrics(self, mini_feature_df):
        from src.models.xgboost_model import XGBoostOilForecaster
        df = mini_feature_df
        feat_cols = [c for c in df.columns
                     if not c.startswith("target_") and c != "close"
                     and c not in ("open", "high", "low", "volume")]
        n = len(df)
        train = df.iloc[:int(n * 0.7)]
        val   = df.iloc[int(n * 0.7):int(n * 0.85)]
        test  = df.iloc[int(n * 0.85):]

        model = XGBoostOilForecaster(reg_params={"n_estimators": 30})
        model.fit(train, val, feat_cols)
        metrics = model.evaluate(test, test["close"])

        for key in ["return_mae", "dir_accuracy", "roc_auc"]:
            assert key in metrics
        assert 0 <= metrics["dir_accuracy"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1


# ─────────────────────────────────────────────────────────────────────────────
# Signal Generator
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalGenerator:

    def test_returns_valid_signal(self, feature_row):
        forecast = {
            "predicted_return":  0.01,
            "direction_prob_up": 0.65,
            "current_price":     82.5,
        }
        result = generate_signal(forecast, feature_row)
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_confidence_in_range(self, feature_row):
        forecast = {"predicted_return": 0.01, "direction_prob_up": 0.70, "current_price": 82.5}
        result = generate_signal(forecast, feature_row)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_high_prob_yields_buy(self, feature_row):
        forecast = {"predicted_return": 0.02, "direction_prob_up": 0.80, "current_price": 82.5}
        result = generate_signal(forecast, feature_row)
        assert result["signal"] == "BUY"

    def test_low_prob_yields_sell(self, feature_row):
        forecast = {"predicted_return": -0.02, "direction_prob_up": 0.25, "current_price": 82.5}
        result = generate_signal(forecast, feature_row)
        assert result["signal"] == "SELL"

    def test_middle_prob_yields_hold(self, feature_row):
        forecast = {"predicted_return": 0.001, "direction_prob_up": 0.51, "current_price": 82.5}
        result = generate_signal(forecast, feature_row)
        assert result["signal"] == "HOLD"

    def test_stop_loss_below_price_for_buy(self, feature_row):
        forecast = {"predicted_return": 0.02, "direction_prob_up": 0.75, "current_price": 82.5}
        result = generate_signal(forecast, feature_row)
        if result["signal"] == "BUY":
            assert result["stop_loss"] < 82.5

    def test_take_profit_above_price_for_buy(self, feature_row):
        forecast = {"predicted_return": 0.02, "direction_prob_up": 0.75, "current_price": 82.5}
        result = generate_signal(forecast, feature_row)
        if result["signal"] == "BUY":
            assert result["take_profit"] > 82.5

    def test_rationale_is_string(self, feature_row):
        forecast = {"predicted_return": 0.01, "direction_prob_up": 0.65, "current_price": 82.5}
        result = generate_signal(forecast, feature_row)
        assert isinstance(result["rationale"], str)
        assert len(result["rationale"]) > 0

    def test_high_vol_reduces_confidence(self):
        """High-volatility regime should reduce signal confidence."""
        row_normal = pd.Series({"rsi_14": 50, "macd_hist": 0.1, "bb_pct_b": 0.5,
                                 "sentiment_score": 0.1, "sentiment_ma5": 0.1,
                                 "eia_weekly_change_mb": -1, "eia_large_draw": 0,
                                 "eia_large_build": 0, "eia_above_seasonal": 0,
                                 "dxy_above_ma": 0, "yield_inverted": 0,
                                 "atr_pct": 0.015, "high_vol_regime": 0,
                                 "roc_5d": 0.01, "ema_cross_50_200": 1,
                                 "extreme_bearish_sentiment": 0, "extreme_bullish_sentiment": 0})
        row_high_vol = row_normal.copy()
        row_high_vol["high_vol_regime"] = 1

        forecast = {"predicted_return": 0.02, "direction_prob_up": 0.70, "current_price": 82.5}
        sig_normal   = generate_signal(forecast, row_normal)
        sig_high_vol = generate_signal(forecast, row_high_vol)

        # High-vol confidence should be ≤ normal confidence
        assert sig_high_vol["confidence"] <= sig_normal["confidence"] + 0.01
