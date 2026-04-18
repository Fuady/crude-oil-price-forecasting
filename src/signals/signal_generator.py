"""
src/signals/signal_generator.py
---------------------------------
BUY / HOLD / SELL signal generation from model forecasts.

Signal logic combines:
  1. Model forecast (direction probability from XGBoost classifier)
  2. Technical confirmation (trend, momentum indicators)
  3. Sentiment filter (don't fight extreme negative sentiment)
  4. Volatility regime filter (reduce position size in high-vol regimes)
  5. Risk management (stop-loss and take-profit levels based on ATR)

Signal confidence scoring (0–1):
  - High (>0.70): strong model agreement + technical confirmation
  - Medium (0.50–0.70): model signal present, mixed technical
  - Low (<0.50): conflicting signals — HOLD recommended

This is the final step between ML model and trading desk action.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from enum import Enum


class Signal(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStrength(str, Enum):
    STRONG   = "STRONG"
    MODERATE = "MODERATE"
    WEAK     = "WEAK"


def generate_signal(
    forecast: Dict,
    current_row: pd.Series,
    thresholds: Optional[Dict] = None,
) -> Dict:
    """
    Generate a BUY/HOLD/SELL signal from model forecasts + technical context.

    Parameters
    ----------
    forecast    : output from EnsembleForecaster.predict_with_ci()
    current_row : latest feature row (contains technical indicators)
    thresholds  : signal threshold overrides (optional)

    Returns
    -------
    dict with: signal, confidence, strength, rationale,
               stop_loss, take_profit, position_size_pct
    """
    th = {
        "buy_prob":   0.60,
        "sell_prob":  0.40,
        "min_conf":   0.45,
        **(thresholds or {}),
    }

    dir_prob = forecast.get("direction_prob_up", 0.5)
    pred_ret = forecast.get("predicted_return", 0.0)
    current_price = forecast.get("current_price",
                                 forecast.get("predicted_price", 80.0))

    # ── Technical confirmation signals ────────────────────────────
    tech_signals = []

    if "rsi_14" in current_row.index:
        rsi = current_row["rsi_14"]
        if rsi < 35:
            tech_signals.append(("RSI oversold", +1))
        elif rsi > 65:
            tech_signals.append(("RSI overbought", -1))

    if "macd_hist" in current_row.index:
        macd_h = current_row["macd_hist"]
        if macd_h > 0:
            tech_signals.append(("MACD bullish histogram", +1))
        else:
            tech_signals.append(("MACD bearish histogram", -1))

    if "ema_cross_50_200" in current_row.index:
        cross = current_row["ema_cross_50_200"]
        if cross == 1:
            tech_signals.append(("Golden cross (EMA50>200)", +1))
        else:
            tech_signals.append(("Death cross (EMA50<200)", -1))

    if "bb_pct_b" in current_row.index:
        pctb = current_row["bb_pct_b"]
        if pctb < 0.2:
            tech_signals.append(("Near Bollinger lower band", +1))
        elif pctb > 0.8:
            tech_signals.append(("Near Bollinger upper band", -1))

    # ── Sentiment filter ──────────────────────────────────────────
    sent_score = current_row.get("sentiment_score", 0.0)
    sent_signals = []

    if sent_score > 0.3:
        sent_signals.append(("Positive news sentiment", +1))
    elif sent_score < -0.3:
        sent_signals.append(("Negative news sentiment", -1))

    if current_row.get("extreme_bearish_sentiment", 0) == 1:
        sent_signals.append(("Extreme bearish sentiment (contrarian buy?)", 0))

    # ── Volatility regime filter ──────────────────────────────────
    high_vol = current_row.get("high_vol_regime", 0) == 1
    atr_pct  = current_row.get("atr_pct", 0.015)

    # ── EIA inventory context ─────────────────────────────────────
    inv_signals = []
    if current_row.get("eia_large_draw", 0) == 1:
        inv_signals.append(("Large EIA crude draw", +1))
    if current_row.get("eia_large_build", 0) == 1:
        inv_signals.append(("Large EIA crude build", -1))
    if current_row.get("eia_above_seasonal", 0) == 0:
        inv_signals.append(("Inventories below seasonal average", +1))

    # ── Macro context ─────────────────────────────────────────────
    macro_signals = []
    if "dxy_above_ma" in current_row.index:
        if current_row["dxy_above_ma"] == 0:
            macro_signals.append(("USD below 20-day MA (oil supportive)", +1))
        else:
            macro_signals.append(("USD above 20-day MA (oil headwind)", -1))

    if "yield_inverted" in current_row.index and current_row["yield_inverted"] == 1:
        macro_signals.append(("Inverted yield curve (recession risk)", -1))

    # ── Aggregate confidence ──────────────────────────────────────
    all_signals   = tech_signals + sent_signals + inv_signals + macro_signals
    bull_count    = sum(1 for _, s in all_signals if s > 0)
    bear_count    = sum(1 for _, s in all_signals if s < 0)
    total_signals = len(all_signals)

    tech_confirmation = (
        (bull_count - bear_count) / max(total_signals, 1)
    )   # range: -1 to +1

    # Blend model probability with technical confirmation
    if dir_prob >= th["buy_prob"]:
        raw_confidence = 0.65 * dir_prob + 0.35 * (0.5 + tech_confirmation * 0.5)
        signal = Signal.BUY
    elif dir_prob <= th["sell_prob"]:
        raw_confidence = 0.65 * (1 - dir_prob) + 0.35 * (0.5 - tech_confirmation * 0.5)
        signal = Signal.SELL
    else:
        raw_confidence = 0.5
        signal = Signal.HOLD

    confidence = float(np.clip(raw_confidence, 0.0, 1.0))

    if confidence < th["min_conf"]:
        signal     = Signal.HOLD
        confidence = 0.5

    # Reduce confidence in high-vol regimes
    if high_vol and signal != Signal.HOLD:
        confidence *= 0.85

    # Signal strength
    if confidence >= 0.72:
        strength = SignalStrength.STRONG
    elif confidence >= 0.58:
        strength = SignalStrength.MODERATE
    else:
        strength = SignalStrength.WEAK

    # ── Risk management levels ────────────────────────────────────
    atr_dollar = current_price * atr_pct
    if signal == Signal.BUY:
        stop_loss   = round(current_price - 2.0 * atr_dollar, 2)
        take_profit = round(current_price + 3.0 * atr_dollar, 2)
    elif signal == Signal.SELL:
        stop_loss   = round(current_price + 2.0 * atr_dollar, 2)
        take_profit = round(current_price - 3.0 * atr_dollar, 2)
    else:
        stop_loss   = round(current_price - 1.5 * atr_dollar, 2)
        take_profit = round(current_price + 1.5 * atr_dollar, 2)

    # Position size (Kelly-inspired, scaled by confidence)
    base_position = 0.10   # 10% base allocation
    position_size = round(base_position * confidence * (1.5 if not high_vol else 0.75), 4)

    # Rationale
    top_bull = [name for name, s in all_signals if s > 0][:3]
    top_bear = [name for name, s in all_signals if s < 0][:2]

    rationale_parts = []
    if top_bull:
        rationale_parts.append("Bullish: " + ", ".join(top_bull))
    if top_bear:
        rationale_parts.append("Bearish: " + ", ".join(top_bear))
    if not rationale_parts:
        rationale_parts.append("Mixed signals — no clear directional bias")
    if high_vol:
        rationale_parts.append("High volatility regime — position size reduced")

    return {
        "signal":           signal.value,
        "confidence":       round(confidence, 4),
        "signal_strength":  strength.value,
        "rationale":        ". ".join(rationale_parts),
        "stop_loss":        stop_loss,
        "take_profit":      take_profit,
        "position_size_pct": position_size,
        "direction_prob_up": round(dir_prob, 4),
        "tech_bull_signals": bull_count,
        "tech_bear_signals": bear_count,
        "high_vol_regime":  bool(high_vol),
        "signal_details":   {
            "technical":  tech_signals,
            "sentiment":  sent_signals,
            "inventory":  inv_signals,
            "macro":      macro_signals,
        },
    }
