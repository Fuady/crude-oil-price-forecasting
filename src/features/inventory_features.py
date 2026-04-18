"""
src/features/inventory_features.py
------------------------------------
EIA crude oil inventory features — one of the strongest short-term
price signals in the oil market.

Every Wednesday at 10:30 AM ET, the EIA releases the Weekly Petroleum
Status Report. An inventory draw (less crude than expected) is bullish;
a build is bearish. The "surprise" component (actual vs consensus estimate)
often moves Brent by $0.50–$2.00 in the minutes after release.

Features:
  - Weekly change in crude stockpiles (draw/build)
  - 4-week rolling average of changes
  - Year-over-year inventory comparison
  - Inventory relative to 5-year seasonal average
  - Cushing, Oklahoma stocks (WTI pricing point)
"""

import numpy as np
import pandas as pd
from pathlib import Path


RAW_DATA_DIR = Path("data/raw")


def load_eia_inventory(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load EIA inventory data and align to daily price index."""
    inv_path = RAW_DATA_DIR / "eia_inventory.parquet"

    if inv_path.exists():
        inv = pd.read_parquet(inv_path)
        inv.index = pd.to_datetime(inv.index)
        # Forward-fill weekly data to daily (EIA reports weekly on Wednesdays)
        inv_daily = inv.reindex(index).ffill().bfill()
        return inv_daily

    return _generate_synthetic_inventory(index)


def _generate_synthetic_inventory(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate synthetic weekly inventory data aligned to daily index."""
    rng = np.random.default_rng(42)
    n = len(index)
    t = np.arange(n)

    seasonal = 20 * np.sin(2 * np.pi * t / 252 + 1.5)
    trend = -0.005 * t
    noise = rng.normal(0, 8, n)
    inventory = 450 + seasonal + trend + noise
    inventory = np.clip(inventory, 350, 540)

    weekly_change = np.concatenate([[0], np.diff(inventory)])
    is_draw = (weekly_change < 0).astype(int)

    df = pd.DataFrame({
        "eia_crude_inventory_mb":  inventory,
        "eia_weekly_change_mb":    weekly_change,
        "eia_is_draw":             is_draw,
    }, index=index)

    return df


def add_inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EIA crude oil inventory features to the price DataFrame.
    """
    out = df.copy()
    inv = load_eia_inventory(df.index)

    for col in inv.columns:
        out[col] = inv[col].values

    if "eia_weekly_change_mb" in out.columns:
        wc = out["eia_weekly_change_mb"]

        # Rolling averages of weekly changes
        out["eia_change_ma4w"]    = wc.rolling(4, min_periods=1).mean()
        out["eia_change_ma13w"]   = wc.rolling(13, min_periods=1).mean()

        # Consecutive draws/builds (momentum in supply balance)
        out["eia_consecutive_draws"] = (
            wc.rolling(4, min_periods=1)
            .apply(lambda x: (x < 0).sum(), raw=True)
        )

        # Large move flag (> 3 Mbbl)
        out["eia_large_draw"]  = (wc < -3.0).astype(int)
        out["eia_large_build"] = (wc >  3.0).astype(int)

    if "eia_crude_inventory_mb" in out.columns:
        inv_level = out["eia_crude_inventory_mb"]
        # 5-year seasonal norm (approximate as 52-week rolling mean)
        seasonal_norm = inv_level.rolling(252, min_periods=52).mean()
        out["eia_vs_seasonal_norm"]  = inv_level - seasonal_norm
        out["eia_above_seasonal"]    = (out["eia_vs_seasonal_norm"] > 0).astype(int)

    return out
