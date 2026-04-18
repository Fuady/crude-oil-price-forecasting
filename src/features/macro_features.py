"""
src/features/macro_features.py
-------------------------------
Macroeconomic and cross-asset features for oil price forecasting.

Key drivers of oil prices beyond technical analysis:
  - USD strength (DXY): inverse correlation — stronger USD → cheaper oil in USD terms
  - Yield curve: slope signals economic growth/recession expectations
  - Inflation expectations: oil is an inflation hedge, CPI drives demand forecasts
  - Equity market: risk-on/risk-off regimes affect commodity demand
  - Gold: safe-haven relationship with oil in geopolitical stress
  - China proxy: copper, CNY — China is the marginal demand driver for oil
  - Geopolitical risk calendar: OPEC meetings, EIA reports, Fed decisions
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


RAW_DATA_DIR = Path("data/raw")


def load_macro_series(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load and align all macro series to the given date index.
    Uses forward-fill for non-trading-day gaps (e.g. monthly CPI → daily).
    """
    frames = {}

    # DXY (USD Index)
    dxy_path = RAW_DATA_DIR / "prices_dxy.parquet"
    if dxy_path.exists():
        dxy = pd.read_parquet(dxy_path)[["close"]].rename(columns={"close": "dxy"})
        frames["dxy"] = dxy

    # Gold
    gold_path = RAW_DATA_DIR / "prices_gold.parquet"
    if gold_path.exists():
        gold = pd.read_parquet(gold_path)[["close"]].rename(columns={"close": "gold"})
        frames["gold"] = gold

    # 10Y yield
    yield_path = RAW_DATA_DIR / "prices_us10y_yield.parquet"
    if yield_path.exists():
        y10 = pd.read_parquet(yield_path)[["close"]].rename(columns={"close": "yield_10y"})
        frames["yield_10y"] = y10

    # FRED macro
    fred_path = RAW_DATA_DIR / "macro_fred.parquet"
    if fred_path.exists():
        fred = pd.read_parquet(fred_path)
        for col in fred.columns:
            frames[col] = fred[[col]]

    if not frames:
        # All-synthetic fallback
        return _generate_synthetic_macro(index)

    combined = pd.concat(frames.values(), axis=1)
    combined = combined.reindex(index).ffill().bfill()
    return combined


def _generate_synthetic_macro(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate synthetic macro data correlated with oil price dynamics."""
    rng = np.random.default_rng(42)
    n = len(index)
    t = np.arange(n)

    df = pd.DataFrame(index=index)
    df["dxy"]       = 98 + rng.normal(0, 3, n) + 0.005 * t   # mild uptrend
    df["gold"]      = 1200 + 0.3 * t + rng.normal(0, 50, n)
    df["yield_10y"] = np.clip(2.5 + rng.normal(0, 0.5, n), 0.1, 5.5)
    df["yield_3m"]  = np.clip(1.5 + rng.normal(0, 0.4, n), 0.0, 5.0)
    df["eurusd"]    = 1.10 + rng.normal(0, 0.05, n)
    df["spx"]       = 2000 + 1.5 * t + rng.normal(0, 100, n)

    return df


def add_macro_features(
    price_df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add macroeconomic features to the price DataFrame.
    """
    out = price_df.copy()

    if macro_df is None:
        macro_df = load_macro_series(price_df.index)

    # Align index
    macro_aligned = macro_df.reindex(price_df.index).ffill().bfill()

    # ── USD Index features ────────────────────────────────────────
    if "dxy" in macro_aligned.columns:
        dxy = macro_aligned["dxy"]
        out["dxy"]          = dxy
        out["dxy_roc_5d"]   = dxy.pct_change(5)
        out["dxy_roc_20d"]  = dxy.pct_change(20)
        out["dxy_ma20"]     = dxy.rolling(20, min_periods=1).mean()
        out["dxy_above_ma"] = (dxy > out["dxy_ma20"]).astype(int)
        # DXY-oil inverse correlation feature
        out["dxy_oil_ratio"] = out["close"] / dxy.clip(lower=1e-6)

    # ── Yield curve features ──────────────────────────────────────
    if "yield_10y" in macro_aligned.columns:
        y10 = macro_aligned["yield_10y"]
        out["yield_10y"]       = y10
        out["yield_10y_roc"]   = y10.diff(5)
        out["yield_10y_ma20"]  = y10.rolling(20, min_periods=1).mean()

    if "yield_10y" in macro_aligned.columns and "yield_3m" in macro_aligned.columns:
        y3m = macro_aligned["yield_3m"]
        out["yield_3m"]        = y3m
        out["yield_curve_slope"] = y10 - y3m   # Positive: normal; Negative: inverted (recession signal)
        out["yield_inverted"]    = (out["yield_curve_slope"] < 0).astype(int)

    # ── Gold features (geopolitical risk proxy) ────────────────────
    if "gold" in macro_aligned.columns:
        gold = macro_aligned["gold"]
        out["gold_price"]      = gold
        out["gold_roc_5d"]     = gold.pct_change(5)
        out["gold_oil_ratio"]  = gold / out["close"].clip(lower=1e-6)   # Risk barometer
        out["gold_oil_roc"]    = out["gold_oil_ratio"].pct_change(5)

    # ── Equity market features ─────────────────────────────────────
    if "spx" in macro_aligned.columns:
        spx = macro_aligned["spx"]
        out["spx_roc_5d"]    = spx.pct_change(5)
        out["spx_roc_20d"]   = spx.pct_change(20)
        out["risk_on_regime"] = (spx.pct_change(20) > 0).astype(int)

    # ── Cross-rate features ───────────────────────────────────────
    if "eurusd" in macro_aligned.columns:
        eurusd = macro_aligned["eurusd"]
        out["eurusd"]        = eurusd
        out["eurusd_roc_5d"] = eurusd.pct_change(5)

    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar and seasonality features.

    Oil markets have strong seasonal patterns:
    - Driving season (May–Aug): higher gasoline demand in US
    - Heating oil season (Oct–Feb): higher distillate demand
    - OPEC meeting months: typically increased volatility
    - EIA report weeks (every Wednesday): intraweek patterns
    """
    out = df.copy()
    idx = df.index

    out["day_of_week"]   = idx.dayofweek           # 0=Mon, 4=Fri
    out["day_of_month"]  = idx.day
    out["month"]         = idx.month
    out["quarter"]       = idx.quarter
    out["year"]          = idx.year

    # Seasonal demand regimes
    out["driving_season"]  = idx.month.isin([5, 6, 7, 8]).astype(int)
    out["heating_season"]  = idx.month.isin([10, 11, 12, 1, 2]).astype(int)

    # OPEC meeting months (typically Dec and Jun for semi-annual meetings)
    out["opec_meeting_month"] = idx.month.isin([6, 12]).astype(int)

    # End-of-quarter / year effects
    out["end_of_quarter"]  = (idx.month.isin([3, 6, 9, 12]) & (idx.day >= 25)).astype(int)

    # Monday effect (oil prices often gap on Monday due to weekend news)
    out["is_monday"] = (idx.dayofweek == 0).astype(int)

    # Sine/cosine encoding for month (captures cyclical seasonality for ML)
    out["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * idx.month / 12)

    return out


def add_lagged_features(
    df: pd.DataFrame,
    target_col: str = "close",
    lags: list = None,
) -> pd.DataFrame:
    """
    Add lagged price features — critical for time-series ML models.
    Lags represent the model's "memory" of past prices.
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10, 20, 60]

    out = df.copy()
    close = df[target_col]

    for lag in lags:
        out[f"lag_{lag}d_close"]  = close.shift(lag)
        out[f"lag_{lag}d_return"] = close.pct_change(lag).shift(1)

    # Target: next-day, 5-day, 20-day forward returns
    out["target_1d_return"]  = close.pct_change(1).shift(-1)    # tomorrow's return
    out["target_5d_return"]  = close.pct_change(5).shift(-5)    # 1-week return
    out["target_20d_return"] = close.pct_change(20).shift(-20)  # 1-month return

    # Direction labels (for classification)
    out["target_direction_1d"]  = (out["target_1d_return"]  > 0).astype(int)
    out["target_direction_5d"]  = (out["target_5d_return"]  > 0).astype(int)
    out["target_direction_20d"] = (out["target_20d_return"] > 0).astype(int)

    return out
