"""
src/features/feature_pipeline.py
----------------------------------
End-to-end feature engineering pipeline.

Orchestrates:
  1. Load raw Brent/WTI price data
  2. Technical indicators (RSI, MACD, Bollinger, ATR, ...)
  3. Macro features (DXY, yield curve, gold, equity)
  4. EIA inventory features (draw/build, seasonal deviation)
  5. NLP sentiment features (FinBERT or lexicon)
  6. Calendar & seasonality features
  7. Lagged price features and targets
  8. Save feature matrix to data/features/

Usage:
    python src/features/feature_pipeline.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.technical_indicators import compute_all_technical_indicators
from src.features.macro_features import add_macro_features, add_calendar_features, add_lagged_features
from src.features.sentiment_analyzer import add_sentiment_features
from src.features.inventory_features import add_inventory_features

RAW_DATA_DIR     = Path("data/raw")
FEATURES_DIR     = Path("data/features")
PROCESSED_DIR    = Path("data/processed")

# Columns to exclude from feature matrix (raw OHLCV + target leakage)
EXCLUDE_COLS = {
    "open", "high", "low", "volume",
    "target_1d_return", "target_5d_return", "target_20d_return",
    "target_direction_1d", "target_direction_5d", "target_direction_20d",
}


def load_price_data(instrument: str = "brent") -> pd.DataFrame:
    """Load OHLCV data for a given instrument."""
    path = RAW_DATA_DIR / f"{instrument}_ohlcv.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Price data not found: {path}\n"
            "Run: python src/ingestion/download_data.py"
        )
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return only feature columns (exclude targets and raw OHLCV)."""
    return [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and c != "close"
        and not c.startswith("target_")
    ]


def run_pipeline(
    instrument: str = "brent",
    test_fraction: float = 0.15,
    val_fraction: float = 0.10,
) -> dict:
    """
    Run the complete feature engineering pipeline.

    Returns
    -------
    dict with:
        train_df, val_df, test_df  — DataFrames with features + targets
        feature_cols               — list of feature column names
        target_cols                — list of target column names
        meta                       — metadata dict for the API
    """
    print(f"\n{'='*60}")
    print(f"  Feature Engineering Pipeline — {instrument.upper()}")
    print(f"{'='*60}")

    # ── 1. Load prices ────────────────────────────────────────────
    print("\n[1/7] Loading price data...")
    price_df = load_price_data(instrument)
    print(f"  Loaded {len(price_df):,} rows  ({price_df.index[0].date()} → {price_df.index[-1].date()})")

    wti_df = None
    if instrument == "brent":
        try:
            wti_df = load_price_data("wti")
        except FileNotFoundError:
            pass

    # ── 2. Technical indicators ───────────────────────────────────
    print("\n[2/7] Computing technical indicators...")
    df = compute_all_technical_indicators(price_df, wti_df)

    # ── 3. Macro features ─────────────────────────────────────────
    print("\n[3/7] Adding macro features...")
    df = add_macro_features(df)
    print(f"  ✓ {sum(1 for c in df.columns if c.startswith(('dxy','yield','gold','spx','eurusd')))} macro features")

    # ── 4. Inventory features ─────────────────────────────────────
    print("\n[4/7] Adding EIA inventory features...")
    df = add_inventory_features(df)
    print(f"  ✓ {sum(1 for c in df.columns if c.startswith('eia_'))} inventory features")

    # ── 5. NLP sentiment ─────────────────────────────────────────
    print("\n[5/7] Adding NLP sentiment features...")
    df = add_sentiment_features(df)
    print(f"  ✓ {sum(1 for c in df.columns if 'sentiment' in c)} sentiment features")

    # ── 6. Calendar & lag features ────────────────────────────────
    print("\n[6/7] Adding calendar & lagged features...")
    df = add_calendar_features(df)
    df = add_lagged_features(df, target_col="close")

    # ── 7. Clean & split ─────────────────────────────────────────
    print("\n[7/7] Cleaning and splitting...")

    # Drop rows at the start with NaN from indicator warm-up
    feature_cols = get_feature_columns(df)
    target_cols  = [c for c in df.columns if c.startswith("target_")]

    # Fill remaining NaNs with forward-fill then zero
    df[feature_cols] = df[feature_cols].ffill().fillna(0)

    # Drop rows where target is NaN (future periods beyond data)
    df = df.dropna(subset=["target_1d_return"])

    n = len(df)
    n_test = int(n * test_fraction)
    n_val  = int(n * val_fraction)
    n_train = n - n_test - n_val

    train_df = df.iloc[:n_train].copy()
    val_df   = df.iloc[n_train:n_train + n_val].copy()
    test_df  = df.iloc[n_train + n_val:].copy()

    print(f"  Train: {len(train_df):,} rows  ({train_df.index[0].date()} → {train_df.index[-1].date()})")
    print(f"  Val:   {len(val_df):,} rows  ({val_df.index[0].date()} → {val_df.index[-1].date()})")
    print(f"  Test:  {len(test_df):,} rows  ({test_df.index[0].date()} → {test_df.index[-1].date()})")
    print(f"\n  Total features: {len(feature_cols)}")
    print(f"  Targets:        {target_cols}")

    # Target statistics
    for tc in target_cols[:3]:
        vals = train_df[tc].dropna()
        pct_up = (vals > 0).mean()
        print(f"    {tc:<30} mean={vals.mean():.4f}  pct_up={pct_up:.1%}")

    # ── Save ─────────────────────────────────────────────────────
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(FEATURES_DIR / f"train_{instrument}.parquet")
    val_df.to_parquet(FEATURES_DIR / f"val_{instrument}.parquet")
    test_df.to_parquet(FEATURES_DIR / f"test_{instrument}.parquet")

    meta = {
        "instrument":   instrument,
        "feature_cols": feature_cols,
        "target_cols":  target_cols,
        "n_features":   len(feature_cols),
        "train_start":  str(train_df.index[0].date()),
        "train_end":    str(train_df.index[-1].date()),
        "test_start":   str(test_df.index[0].date()),
        "test_end":     str(test_df.index[-1].date()),
        "last_price":   float(df["close"].iloc[-1]),
        "last_date":    str(df.index[-1].date()),
    }
    joblib.dump(meta, FEATURES_DIR / f"meta_{instrument}.joblib")

    print(f"\n✓ Features saved to {FEATURES_DIR.resolve()}")

    return {
        "train_df":    train_df,
        "val_df":      val_df,
        "test_df":     test_df,
        "full_df":     df,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "meta":        meta,
    }


def load_features(instrument: str = "brent") -> dict:
    """Load pre-computed feature matrices from disk."""
    paths = {
        "train": FEATURES_DIR / f"train_{instrument}.parquet",
        "val":   FEATURES_DIR / f"val_{instrument}.parquet",
        "test":  FEATURES_DIR / f"test_{instrument}.parquet",
        "meta":  FEATURES_DIR / f"meta_{instrument}.joblib",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Features not found ({missing}). "
            "Run: python src/features/feature_pipeline.py"
        )

    meta = joblib.load(paths["meta"])
    return {
        "train_df":    pd.read_parquet(paths["train"]),
        "val_df":      pd.read_parquet(paths["val"]),
        "test_df":     pd.read_parquet(paths["test"]),
        "feature_cols": meta["feature_cols"],
        "target_cols": meta["target_cols"],
        "meta":        meta,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default="brent",
                        choices=["brent", "wti"])
    parser.add_argument("--both", action="store_true",
                        help="Process both Brent and WTI")
    args = parser.parse_args()

    instruments = ["brent", "wti"] if args.both else [args.instrument]
    for inst in instruments:
        try:
            run_pipeline(inst)
        except FileNotFoundError as e:
            print(f"\n{e}")

    print("\n✓ Feature pipeline complete.")
    print("  Next: python src/models/train.py")
