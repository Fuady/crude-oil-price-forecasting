"""
tests/test_features.py
------------------------
Unit tests for technical indicators and feature engineering.
Run: pytest tests/test_features.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.technical_indicators import (
    add_moving_averages, add_macd, add_rsi, add_bollinger_bands,
    add_atr, add_rate_of_change, add_historical_volatility,
    add_volume_indicators, add_support_resistance,
)
from src.features.macro_features import add_calendar_features, add_lagged_features


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv():
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 300
    dates  = pd.bdate_range("2022-01-01", periods=n)
    close  = 80 + np.cumsum(np.random.normal(0, 1.2, n))
    close  = np.clip(close, 30, 150)
    high   = close * (1 + np.random.uniform(0.001, 0.012, n))
    low    = close * (1 - np.random.uniform(0.001, 0.012, n))
    open_  = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.randint(100000, 500000, n)

    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMovingAverages:

    def test_adds_sma_columns(self, sample_ohlcv):
        result = add_moving_averages(sample_ohlcv)
        for period in [5, 10, 20, 50, 200]:
            assert f"sma_{period}" in result.columns

    def test_adds_ema_columns(self, sample_ohlcv):
        result = add_moving_averages(sample_ohlcv)
        for period in [5, 10, 20, 50, 200]:
            assert f"ema_{period}" in result.columns

    def test_sma_20_equals_rolling_mean(self, sample_ohlcv):
        result = add_moving_averages(sample_ohlcv)
        expected = sample_ohlcv["close"].rolling(20, min_periods=1).mean()
        pd.testing.assert_series_equal(
            result["sma_20"].round(8), expected.round(8), check_names=False
        )

    def test_golden_cross_binary(self, sample_ohlcv):
        result = add_moving_averages(sample_ohlcv)
        assert set(result["ema_cross_50_200"].unique()).issubset({0, 1})

    def test_price_vs_sma_range(self, sample_ohlcv):
        result = add_moving_averages(sample_ohlcv)
        # Ratio should be close to 0 on average (not hundreds)
        assert result["price_vs_sma20"].abs().median() < 0.5


class TestRSI:

    def test_rsi_range(self, sample_ohlcv):
        result = add_rsi(sample_ohlcv)
        valid = result["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_oversold_overbought_binary(self, sample_ohlcv):
        result = add_rsi(sample_ohlcv)
        assert set(result["rsi_oversold"].unique()).issubset({0, 1})
        assert set(result["rsi_overbought"].unique()).issubset({0, 1})

    def test_rsi_not_both_oversold_and_overbought(self, sample_ohlcv):
        result = add_rsi(sample_ohlcv)
        both = (result["rsi_oversold"] == 1) & (result["rsi_overbought"] == 1)
        assert not both.any()


class TestMACD:

    def test_adds_macd_columns(self, sample_ohlcv):
        result = add_macd(sample_ohlcv)
        for col in ["macd_line", "macd_signal", "macd_hist"]:
            assert col in result.columns

    def test_macd_hist_is_difference(self, sample_ohlcv):
        result = add_macd(sample_ohlcv)
        expected = result["macd_line"] - result["macd_signal"]
        pd.testing.assert_series_equal(
            result["macd_hist"].round(8), expected.round(8), check_names=False
        )


class TestBollingerBands:

    def test_adds_bb_columns(self, sample_ohlcv):
        result = add_bollinger_bands(sample_ohlcv)
        for col in ["bb_upper", "bb_lower", "bb_middle", "bb_pct_b", "bb_width"]:
            assert col in result.columns

    def test_upper_above_lower(self, sample_ohlcv):
        result = add_bollinger_bands(sample_ohlcv)
        assert (result["bb_upper"] >= result["bb_lower"]).all()

    def test_bb_width_nonnegative(self, sample_ohlcv):
        result = add_bollinger_bands(sample_ohlcv)
        assert (result["bb_width"] >= 0).all()

    def test_bb_pct_b_near_zero_when_price_below_lower(self, sample_ohlcv):
        result = add_bollinger_bands(sample_ohlcv)
        below_lower = result[result["bb_below_lower"] == 1]
        if len(below_lower) > 0:
            assert (below_lower["bb_pct_b"] <= 0.1).all()


class TestATR:

    def test_atr_nonnegative(self, sample_ohlcv):
        result = add_atr(sample_ohlcv)
        assert (result["atr_14"] >= 0).all()

    def test_atr_pct_reasonable_range(self, sample_ohlcv):
        result = add_atr(sample_ohlcv)
        valid = result["atr_pct"].dropna()
        # ATR % should be between 0.1% and 10% for oil
        assert (valid >= 0).all()
        assert (valid < 0.20).all()


class TestRateOfChange:

    def test_roc_columns_added(self, sample_ohlcv):
        result = add_rate_of_change(sample_ohlcv)
        for period in [1, 5, 10, 20]:
            assert f"roc_{period}d" in result.columns

    def test_log_returns_finite(self, sample_ohlcv):
        result = add_rate_of_change(sample_ohlcv)
        valid = result["log_return_1d"].dropna()
        assert np.isfinite(valid).all()

    def test_roc_1d_consistent_with_pct_change(self, sample_ohlcv):
        result = add_rate_of_change(sample_ohlcv)
        expected = sample_ohlcv["close"].pct_change(1)
        pd.testing.assert_series_equal(
            result["roc_1d"].round(8), expected.round(8), check_names=False
        )


class TestHistoricalVolatility:

    def test_hvol_nonnegative(self, sample_ohlcv):
        result = add_historical_volatility(sample_ohlcv)
        for period in [10, 20, 60]:
            col = f"hvol_{period}d"
            assert col in result.columns
            valid = result[col].dropna()
            assert (valid >= 0).all()

    def test_high_vol_regime_binary(self, sample_ohlcv):
        result = add_historical_volatility(sample_ohlcv)
        assert set(result["high_vol_regime"].unique()).issubset({0, 1})


class TestCalendarFeatures:

    def test_calendar_columns_added(self, sample_ohlcv):
        result = add_calendar_features(sample_ohlcv)
        for col in ["day_of_week", "month", "quarter",
                    "driving_season", "heating_season", "month_sin", "month_cos"]:
            assert col in result.columns

    def test_month_encoding_range(self, sample_ohlcv):
        result = add_calendar_features(sample_ohlcv)
        assert (result["month_sin"].abs() <= 1.0).all()
        assert (result["month_cos"].abs() <= 1.0).all()

    def test_season_binary(self, sample_ohlcv):
        result = add_calendar_features(sample_ohlcv)
        assert set(result["driving_season"].unique()).issubset({0, 1})
        assert set(result["heating_season"].unique()).issubset({0, 1})


class TestLaggedFeatures:

    def test_lag_columns_added(self, sample_ohlcv):
        result = add_lagged_features(sample_ohlcv)
        for lag in [1, 5, 10]:
            assert f"lag_{lag}d_close" in result.columns

    def test_targets_created(self, sample_ohlcv):
        result = add_lagged_features(sample_ohlcv)
        for col in ["target_1d_return", "target_5d_return", "target_direction_1d"]:
            assert col in result.columns

    def test_target_direction_binary(self, sample_ohlcv):
        result = add_lagged_features(sample_ohlcv)
        valid = result["target_direction_1d"].dropna()
        assert set(valid.unique()).issubset({0, 1})
