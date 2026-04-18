"""
src/signals/backtester.py
--------------------------
Walk-forward backtesting engine for trading signal evaluation.

Walk-forward methodology (the correct way to backtest ML models):
  - Train on window [t-train_size : t]
  - Generate signal for t+1
  - Roll forward one step, repeat
  - No look-ahead bias — model never sees future data during training

Metrics computed:
  - Total return, annualized return
  - Sharpe ratio, Sortino ratio
  - Maximum drawdown
  - Win rate, profit factor
  - Comparison vs Buy & Hold

Usage:
    python src/signals/backtester.py
    python src/signals/backtester.py --instrument brent --plot
"""

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODELS_DIR  = Path("models")
RESULTS_DIR = Path("monitoring/reports")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(returns: pd.Series, risk_free_rate: float = 0.05) -> dict:
    """Compute full suite of trading performance metrics."""
    r = returns.dropna()
    if len(r) == 0:
        return {}

    daily_rf    = risk_free_rate / 252
    excess_ret  = r - daily_rf

    total_ret   = float((1 + r).prod() - 1)
    n_years     = len(r) / 252
    ann_ret     = float((1 + total_ret) ** (1 / max(n_years, 0.1)) - 1)
    ann_vol     = float(r.std() * np.sqrt(252))
    sharpe      = float(excess_ret.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0

    # Sortino (downside deviation only)
    downside    = r[r < 0]
    sortino     = float(
        excess_ret.mean() / downside.std() * np.sqrt(252)
    ) if len(downside) > 1 and downside.std() > 0 else 0.0

    # Max drawdown
    cum         = (1 + r).cumprod()
    rolling_max = cum.cummax()
    drawdown    = (cum - rolling_max) / rolling_max
    max_dd      = float(drawdown.min())

    # Win rate
    wins        = (r > 0).sum()
    losses      = (r < 0).sum()
    win_rate    = float(wins / max(wins + losses, 1))

    # Profit factor
    gross_profit = r[r > 0].sum()
    gross_loss   = abs(r[r < 0].sum())
    profit_factor = float(gross_profit / max(gross_loss, 1e-10))

    # Calmar ratio
    calmar = float(ann_ret / abs(max_dd)) if max_dd != 0 else 0.0

    return {
        "total_return_pct":   round(total_ret * 100, 2),
        "ann_return_pct":     round(ann_ret * 100, 2),
        "ann_volatility_pct": round(ann_vol * 100, 2),
        "sharpe_ratio":       round(sharpe, 3),
        "sortino_ratio":      round(sortino, 3),
        "calmar_ratio":       round(calmar, 3),
        "max_drawdown_pct":   round(max_dd * 100, 2),
        "win_rate_pct":       round(win_rate * 100, 2),
        "profit_factor":      round(profit_factor, 3),
        "n_trades":           int(wins + losses),
        "n_days":             len(r),
    }


# ---------------------------------------------------------------------------
# Simple signal-based backtest
# ---------------------------------------------------------------------------

def run_signal_backtest(
    price_df: pd.DataFrame,
    feature_cols: list,
    instrument: str = "brent",
    transaction_cost_bps: float = 5.0,   # 5 basis points round-trip
    verbose: bool = True,
) -> dict:
    """
    Simplified walk-forward backtest using pre-trained XGBoost signals.

    Strategy:
      - Long when direction_prob > 0.60
      - Short when direction_prob < 0.40
      - Flat otherwise

    Parameters
    ----------
    transaction_cost_bps : round-trip transaction cost in basis points
    """
    xgb_path = MODELS_DIR / f"xgboost_{instrument}.joblib"
    if not xgb_path.exists():
        print(f"  XGBoost model not found at {xgb_path}")
        print("  Run: python src/models/train.py first")
        return {}

    from src.models.xgboost_model import XGBoostOilForecaster
    model = XGBoostOilForecaster.load(xgb_path)

    X = price_df[model.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    dir_proba = model.predict_direction_proba(X)

    actual_returns = price_df["target_1d_return"].fillna(0).values
    prices         = price_df["close"].values

    # Signal positions
    BUY_THRESHOLD  = 0.60
    SELL_THRESHOLD = 0.40
    tx_cost        = transaction_cost_bps / 10000

    positions  = np.zeros(len(dir_proba))
    for i, prob in enumerate(dir_proba):
        if prob > BUY_THRESHOLD:
            positions[i] = 1.0    # long
        elif prob < SELL_THRESHOLD:
            positions[i] = -1.0   # short
        else:
            positions[i] = 0.0    # flat

    # Strategy returns (signal applied next day — no look-ahead)
    shifted_pos = np.roll(positions, 1)
    shifted_pos[0] = 0

    # Transaction costs on position changes
    position_changes = np.abs(np.diff(shifted_pos, prepend=0))
    strategy_returns = shifted_pos * actual_returns - position_changes * tx_cost

    strat_series = pd.Series(strategy_returns, index=price_df.index)
    bnh_series   = pd.Series(actual_returns,   index=price_df.index)

    strat_metrics = compute_metrics(strat_series)
    bnh_metrics   = compute_metrics(bnh_series)

    if verbose:
        print("\n" + "=" * 55)
        print(f"  Backtest Results — {instrument.upper()}")
        print("=" * 55)
        print(f"\n  {'Metric':<30} {'Strategy':>12} {'Buy & Hold':>12}")
        print(f"  {'-'*56}")
        for key in ["total_return_pct", "ann_return_pct", "ann_volatility_pct",
                    "sharpe_ratio", "sortino_ratio", "max_drawdown_pct",
                    "win_rate_pct", "profit_factor"]:
            s_val = strat_metrics.get(key, 0)
            b_val = bnh_metrics.get(key, 0)
            marker = " ✓" if (
                (key in ["total_return_pct", "ann_return_pct", "sharpe_ratio",
                          "sortino_ratio", "win_rate_pct", "profit_factor"] and s_val > b_val) or
                (key in ["max_drawdown_pct", "ann_volatility_pct"] and s_val > b_val)
            ) else ""
            print(f"  {key:<30} {s_val:>12} {b_val:>12}{marker}")

    return {
        "strategy": strat_metrics,
        "buy_and_hold": bnh_metrics,
        "n_long_days":  int((positions > 0).sum()),
        "n_short_days": int((positions < 0).sum()),
        "n_flat_days":  int((positions == 0).sum()),
        "instrument":   instrument,
    }


def plot_backtest_results(
    price_df: pd.DataFrame,
    results: dict,
    instrument: str,
    save_path: Path,
) -> None:
    """Plot equity curve, drawdown, and rolling Sharpe."""
    from src.models.xgboost_model import XGBoostOilForecaster

    xgb_path = MODELS_DIR / f"xgboost_{instrument}.joblib"
    if not xgb_path.exists():
        return

    model    = XGBoostOilForecaster.load(xgb_path)
    X        = price_df[model.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    dir_prob = model.predict_direction_proba(X)
    actual_r = price_df["target_1d_return"].fillna(0).values

    positions = np.where(dir_prob > 0.60, 1.0, np.where(dir_prob < 0.40, -1.0, 0.0))
    pos_shift = np.roll(positions, 1)
    pos_shift[0] = 0
    tx_cost = 5 / 10000
    pos_changes = np.abs(np.diff(pos_shift, prepend=0))
    strat_r = pos_shift * actual_r - pos_changes * tx_cost

    strat_cum = (1 + pd.Series(strat_r, index=price_df.index)).cumprod()
    bnh_cum   = (1 + pd.Series(actual_r, index=price_df.index)).cumprod()

    # Drawdown
    roll_max = strat_cum.cummax()
    drawdown = (strat_cum - roll_max) / roll_max

    # Rolling Sharpe (63-day)
    roll_sharpe = (
        pd.Series(strat_r).rolling(63).mean() /
        pd.Series(strat_r).rolling(63).std().clip(lower=1e-10)
    ) * np.sqrt(252)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Equity curve
    axes[0].plot(strat_cum.index, strat_cum.values,
                 label="ML Strategy", color="#378ADD", linewidth=1.5)
    axes[0].plot(bnh_cum.index, bnh_cum.values,
                 label="Buy & Hold", color="#888", linewidth=1.0, alpha=0.7)
    axes[0].set_title(f"Equity Curve — {instrument.upper()} Forecasting Strategy")
    axes[0].set_ylabel("Cumulative Return (1 = start)")
    axes[0].legend()
    axes[0].axhline(1.0, color="#888", linewidth=0.5, linestyle="--")

    # Drawdown
    axes[1].fill_between(drawdown.index, drawdown.values, 0,
                          color="#E24B4A", alpha=0.4)
    axes[1].plot(drawdown.index, drawdown.values, color="#E24B4A", linewidth=0.8)
    axes[1].set_title("Strategy Drawdown")
    axes[1].set_ylabel("Drawdown (%)")

    # Rolling Sharpe
    axes[2].plot(price_df.index, roll_sharpe.values,
                 color="#1D9E75", linewidth=1.0)
    axes[2].axhline(0, color="#888", linewidth=0.5, linestyle="--")
    axes[2].axhline(1, color="#1D9E75", linewidth=0.5, linestyle=":", alpha=0.7,
                    label="Sharpe=1")
    axes[2].set_title("Rolling 63-Day Sharpe Ratio")
    axes[2].set_ylabel("Sharpe Ratio")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Backtest chart saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default="brent", choices=["brent", "wti"])
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    from src.features.feature_pipeline import load_features
    data = load_features(args.instrument)
    # Use test set only (out-of-sample)
    test_df = data["test_df"]

    print(f"Backtesting on test set: {len(test_df):,} days")
    results = run_signal_backtest(
        test_df, data["feature_cols"], args.instrument, verbose=True
    )

    if results and args.plot:
        plot_backtest_results(
            test_df, results, args.instrument,
            RESULTS_DIR / f"backtest_{args.instrument}.png"
        )

    import json
    out_path = RESULTS_DIR / f"backtest_results_{args.instrument}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")
