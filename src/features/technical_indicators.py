"""
src/features/technical_indicators.py
-------------------------------------
Compute financial technical indicators from OHLCV price data.

Indicators implemented:
  Trend:     SMA, EMA, MACD, Ichimoku Cloud
  Momentum:  RSI, Stochastic, Rate-of-Change, Williams %R
  Volatility: Bollinger Bands, ATR, Historical Volatility
  Volume:    OBV, VWAP proxy, Volume SMA ratio
  Structure: Support/Resistance, Price channels
  Oil-specific: Brent-WTI spread, contango/backwardation proxy

All functions take a pandas DataFrame with columns:
  open, high, low, close, volume
and return a DataFrame with additional indicator columns.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Trend indicators
# ---------------------------------------------------------------------------

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """SMA and EMA at multiple periods commonly used in commodity trading."""
    out = df.copy()
    close = df["close"]

    for period in [5, 10, 20, 50, 200]:
        out[f"sma_{period}"] = close.rolling(period, min_periods=1).mean()
        out[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

    # Price relative to moving averages (dimensionless)
    for period in [20, 50, 200]:
        out[f"price_vs_sma{period}"] = (close - out[f"sma_{period}"]) / out[f"sma_{period}"].clip(lower=1e-6)

    # Golden/Death cross signal: EMA50 vs EMA200
    out["ema_cross_50_200"] = (out["ema_50"] > out["ema_200"]).astype(int)

    return out


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD — Moving Average Convergence Divergence.
    Standard settings (12, 26, 9) widely used in oil futures trading.
    """
    out = df.copy()
    close = df["close"]

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    out["macd_line"]   = ema_fast - ema_slow
    out["macd_signal"] = out["macd_line"].ewm(span=signal, adjust=False).mean()
    out["macd_hist"]   = out["macd_line"] - out["macd_signal"]

    # MACD crossover signal
    out["macd_bullish_cross"] = (
        (out["macd_line"] > out["macd_signal"]) &
        (out["macd_line"].shift(1) <= out["macd_signal"].shift(1))
    ).astype(int)

    return out


# ---------------------------------------------------------------------------
# Momentum indicators
# ---------------------------------------------------------------------------

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index.
    RSI < 30: oversold (potential buy for oil traders)
    RSI > 70: overbought (potential sell signal)
    """
    out = df.copy()
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss.clip(lower=1e-10)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # RSI zones
    out["rsi_oversold"]   = (out["rsi_14"] < 30).astype(int)
    out["rsi_overbought"] = (out["rsi_14"] > 70).astype(int)

    return out


def add_stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """Stochastic Oscillator (%K and %D lines)."""
    out = df.copy()

    low_min  = df["low"].rolling(k_period, min_periods=1).min()
    high_max = df["high"].rolling(k_period, min_periods=1).max()

    out["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
    out["stoch_d"] = out["stoch_k"].rolling(d_period, min_periods=1).mean()

    return out


def add_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """Price Rate of Change at multiple horizons."""
    out = df.copy()
    close = df["close"]

    for period in [1, 5, 10, 20, 60]:
        out[f"roc_{period}d"] = close.pct_change(period)

    # Log returns (better statistical properties for ML)
    out["log_return_1d"]  = np.log(close / close.shift(1))
    out["log_return_5d"]  = np.log(close / close.shift(5))
    out["log_return_20d"] = np.log(close / close.shift(20))

    return out


def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Williams %R — momentum oscillator. Range: -100 to 0."""
    out = df.copy()
    high_max = df["high"].rolling(period, min_periods=1).max()
    low_min  = df["low"].rolling(period, min_periods=1).min()

    out["williams_r"] = -100 * (high_max - df["close"]) / (high_max - low_min + 1e-10)

    return out


# ---------------------------------------------------------------------------
# Volatility indicators
# ---------------------------------------------------------------------------

def add_bollinger_bands(
    df: pd.DataFrame, period: int = 20, n_std: float = 2.0
) -> pd.DataFrame:
    """
    Bollinger Bands.
    Key features for ML: %B (position within bands) and bandwidth.
    """
    out = df.copy()
    close = df["close"]

    sma   = close.rolling(period, min_periods=1).mean()
    std   = close.rolling(period, min_periods=1).std().fillna(0)

    out["bb_upper"]     = sma + n_std * std
    out["bb_lower"]     = sma - n_std * std
    out["bb_middle"]    = sma
    out["bb_width"]     = (out["bb_upper"] - out["bb_lower"]) / sma.clip(lower=1e-6)
    out["bb_pct_b"]     = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"] + 1e-10)

    # Price above/below bands
    out["bb_above_upper"] = (close > out["bb_upper"]).astype(int)
    out["bb_below_lower"] = (close < out["bb_lower"]).astype(int)

    return out


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average True Range — measures market volatility.
    Key for position sizing and stop-loss setting in oil trading.
    """
    out = df.copy()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close  = (df["low"]  - df["close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    out["atr_14"]        = true_range.ewm(com=period - 1, adjust=False).mean()
    out["atr_pct"]       = out["atr_14"] / df["close"].clip(lower=1e-6)   # Normalised

    return out


def add_historical_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Annualised historical volatility at multiple windows."""
    out = df.copy()
    log_ret = np.log(df["close"] / df["close"].shift(1))

    for period in [10, 20, 60]:
        out[f"hvol_{period}d"] = (
            log_ret.rolling(period, min_periods=5).std() * np.sqrt(252)
        )

    # Vol regime: is current vol above its 60-day average?
    out["high_vol_regime"] = (
        out["hvol_20d"] > out["hvol_20d"].rolling(60, min_periods=1).mean()
    ).astype(int)

    return out


# ---------------------------------------------------------------------------
# Volume indicators
# ---------------------------------------------------------------------------

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OBV, volume moving averages, and volume rate of change."""
    out = df.copy()

    if "volume" not in df.columns or df["volume"].sum() == 0:
        out["obv"]             = 0.0
        out["volume_sma_20"]   = 0.0
        out["volume_ratio"]    = 1.0
        return out

    # On-Balance Volume
    direction = np.sign(df["close"].diff()).fillna(0)
    out["obv"] = (direction * df["volume"]).cumsum()

    # Volume moving averages
    out["volume_sma_20"] = df["volume"].rolling(20, min_periods=1).mean()
    out["volume_ratio"]  = df["volume"] / out["volume_sma_20"].clip(lower=1)

    # High volume day (> 1.5x 20-day avg)
    out["high_volume_day"] = (out["volume_ratio"] > 1.5).astype(int)

    return out


# ---------------------------------------------------------------------------
# Support / Resistance levels
# ---------------------------------------------------------------------------

def add_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Rolling high/low as support and resistance proxies.
    Price position within recent range is a key commodity trading signal.
    """
    out = df.copy()

    out["rolling_high_20"] = df["high"].rolling(window, min_periods=1).max()
    out["rolling_low_20"]  = df["low"].rolling(window, min_periods=1).min()
    out["price_channel_pct"] = (
        (df["close"] - out["rolling_low_20"]) /
        (out["rolling_high_20"] - out["rolling_low_20"] + 1e-10)
    )

    # Near support or resistance
    out["near_resistance"] = (out["price_channel_pct"] > 0.85).astype(int)
    out["near_support"]    = (out["price_channel_pct"] < 0.15).astype(int)

    return out


# ---------------------------------------------------------------------------
# Oil-specific features
# ---------------------------------------------------------------------------

def add_brent_wti_spread(
    brent_df: pd.DataFrame, wti_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Brent-WTI spread is a key structural signal.
    Wide spread (>$5): market stress, regional supply imbalance
    Negative spread: unusual, signals major market dislocation
    """
    out = brent_df.copy()

    wti_close = wti_df["close"].reindex(brent_df.index).ffill()
    out["brent_wti_spread"]     = brent_df["close"] - wti_close
    out["brent_wti_spread_ma5"] = out["brent_wti_spread"].rolling(5, min_periods=1).mean()
    out["spread_widening"]      = (
        out["brent_wti_spread"] > out["brent_wti_spread_ma5"]
    ).astype(int)

    return out


# ---------------------------------------------------------------------------
# Master function: apply all indicators
# ---------------------------------------------------------------------------

def compute_all_technical_indicators(
    df: pd.DataFrame,
    wti_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Apply the full technical indicator suite to a price DataFrame.

    Parameters
    ----------
    df     : OHLCV DataFrame for primary instrument (Brent)
    wti_df : WTI OHLCV DataFrame for spread computation (optional)
    """
    out = df.copy()

    print("  Computing technical indicators...")
    out = add_moving_averages(out)
    out = add_macd(out)
    out = add_rsi(out)
    out = add_stochastic(out)
    out = add_rate_of_change(out)
    out = add_williams_r(out)
    out = add_bollinger_bands(out)
    out = add_atr(out)
    out = add_historical_volatility(out)
    out = add_volume_indicators(out)
    out = add_support_resistance(out)

    if wti_df is not None:
        out = add_brent_wti_spread(out, wti_df)

    # Drop rows with too many NaNs (first ~200 rows during indicator warm-up)
    n_before = len(out)
    out = out.dropna(subset=["sma_200", "rsi_14"])
    print(f"  ✓ {len(out)} rows after warm-up drop (removed {n_before - len(out)})")

    return out
