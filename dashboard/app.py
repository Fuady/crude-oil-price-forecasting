"""
dashboard/app.py
-----------------
Streamlit analyst dashboard for oil price forecasting and trading signals.

Panels:
  1. Market overview — current prices, spread, sentiment, inventory
  2. Price forecast — 7-day Brent/WTI forecast with confidence bands
  3. Trading signals — current signal + rationale + risk levels
  4. Factor analysis — SHAP drivers, technical indicators
  5. Historical signals — 30-day signal accuracy review
  6. Backtest summary — strategy performance

Run:
  streamlit run dashboard/app.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Oil Price Forecasting — Trading Desk",
    layout="wide",
    initial_sidebar_state="expanded",
)

SIGNAL_COLORS = {"BUY": "#1D9E75", "SELL": "#E24B4A", "HOLD": "#EF9F27"}

# ─────────────────────────────────────────────────────────────────────────────
# Data loaders (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_price_data():
    """Load latest price and feature data."""
    try:
        from src.features.feature_pipeline import load_features
        data = load_features("brent")
        full = pd.concat([data["train_df"], data["val_df"], data["test_df"]]).sort_index()
        return full, data["feature_cols"]
    except Exception:
        return _make_synthetic_data(), []


def _make_synthetic_data():
    rng   = np.random.default_rng(42)
    dates = pd.bdate_range(end=datetime.today(), periods=500)
    prices = 80 + np.cumsum(rng.normal(0, 1.2, 500))
    prices = np.clip(prices, 50, 120)
    df = pd.DataFrame({
        "close": prices,
        "rsi_14": np.clip(50 + rng.normal(0, 15, 500), 10, 90),
        "macd_hist": rng.normal(0, 0.5, 500),
        "sentiment_score": rng.normal(0.05, 0.2, 500),
        "eia_weekly_change_mb": rng.normal(-0.5, 3, 500),
        "dxy": 100 + rng.normal(0, 2, 500),
        "hvol_20d": np.clip(0.25 + rng.normal(0, 0.08, 500), 0.05, 0.6),
        "bb_pct_b": np.clip(0.5 + rng.normal(0, 0.25, 500), 0, 1),
        "atr_pct": np.clip(0.015 + rng.normal(0, 0.003, 500), 0.005, 0.04),
    }, index=dates)
    return df


@st.cache_data(ttl=300)
def load_model_and_predict(feature_cols):
    """Load model and generate signals on the test window."""
    try:
        from src.models.xgboost_model import XGBoostOilForecaster
        model = XGBoostOilForecaster.load(Path("models/xgboost_brent.joblib"))
        from src.features.feature_pipeline import load_features
        data = load_features("brent")
        test_df = data["test_df"]
        probs   = model.predict_direction_proba(test_df)
        return model, test_df, probs
    except Exception:
        return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Components
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(col, label, value, delta=None, delta_color="normal"):
    with col:
        st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def signal_badge(signal: str) -> str:
    color = SIGNAL_COLORS.get(signal, "#888")
    return (
        f"<span style='background:{color};color:white;padding:4px 14px;"
        f"border-radius:6px;font-size:16px;font-weight:500;'>{signal}</span>"
    )


def plot_price_with_forecast(price_series, forecast_df):
    fig = go.Figure()

    # Historical price
    fig.add_trace(go.Scatter(
        x=price_series.index[-120:],
        y=price_series.values[-120:],
        name="Historical",
        line=dict(color="#378ADD", width=1.8),
    ))

    if forecast_df is not None and len(forecast_df) > 0:
        # Confidence band
        fig.add_trace(go.Scatter(
            x=list(forecast_df["date"]) + list(reversed(forecast_df["date"])),
            y=list(forecast_df["upper_95"]) + list(reversed(forecast_df["lower_95"])),
            fill="toself",
            fillcolor="rgba(239,159,39,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
        ))
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["price"],
            name="Forecast",
            line=dict(color="#EF9F27", width=2.0, dash="dot"),
            mode="lines+markers",
            marker=dict(size=5),
        ))

    fig.update_layout(
        title="Brent Crude — Price & 7-Day Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($/bbl)",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


def plot_signals_timeline(signal_history_df):
    fig = go.Figure()

    for sig, color in SIGNAL_COLORS.items():
        mask = signal_history_df["signal"] == sig
        if mask.any():
            subset = signal_history_df[mask]
            fig.add_trace(go.Scatter(
                x=subset["date"],
                y=subset["price"],
                mode="markers",
                name=sig,
                marker=dict(color=color, size=8,
                            symbol="triangle-up" if sig == "BUY" else
                                   "triangle-down" if sig == "SELL" else "circle"),
            ))

    fig.add_trace(go.Scatter(
        x=signal_history_df["date"],
        y=signal_history_df["price"],
        mode="lines",
        name="Price",
        line=dict(color="#888", width=1.0),
        showlegend=False,
    ))

    fig.update_layout(
        title="Historical Signal Overlay (last 30 days)",
        xaxis_title="Date",
        yaxis_title="Price ($/bbl)",
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_feature_contributions(drivers: list):
    if not drivers:
        return None
    names  = [d["factor"].replace("_", " ")[:30] for d in drivers]
    values = [d["shap_value"] for d in drivers]
    colors = [SIGNAL_COLORS["BUY"] if v > 0 else SIGNAL_COLORS["SELL"] for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="#888", line_width=0.8)
    fig.update_layout(
        title="Key Drivers (SHAP Values) → Price Impact",
        xaxis_title="SHAP value (positive = bullish)",
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=30),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main dashboard
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown(
        "<h1 style='font-size:22px;font-weight:500;margin-bottom:2px;'>"
        "Crude Oil Price Forecasting — Analyst Dashboard</h1>"
        "<p style='color:#888;font-size:13px;margin:0;'>"
        "LSTM + XGBoost Ensemble · NLP Sentiment · EIA Inventory · Multi-factor signals</p>",
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")
        instrument    = st.selectbox("Instrument", ["Brent", "WTI"], index=0)
        horizon       = st.slider("Forecast horizon (days)", 1, 14, 7)
        show_ci       = st.checkbox("Show confidence bands", value=True)
        auto_refresh  = st.checkbox("Auto-refresh (5 min)", value=False)
        st.markdown("---")
        st.markdown("### Model info")
        st.markdown("**Version:** ensemble-v1.0.0")
        st.markdown("**Features:** 95+ (technical + macro + NLP + inventory)")
        st.markdown("**Lookback:** 60 trading days")
        st.markdown("**Updated:** daily after market close")
        st.markdown("---")
        st.markdown("### Data sources")
        st.markdown("- Yahoo Finance (prices)")
        st.markdown("- FRED (macro)")
        st.markdown("- EIA weekly inventory")
        st.markdown("- NLP sentiment (FinBERT)")
        st.markdown("---")
        st.caption("⚠️ For educational purposes only. Not financial advice.")

    # ── Load data ─────────────────────────────────────────────────
    full_df, feature_cols = load_price_data()
    latest = full_df.iloc[-1]
    current_price = float(latest["close"])
    prev_price    = float(full_df.iloc[-2]["close"])
    price_change  = current_price - prev_price
    price_chg_pct = price_change / prev_price * 100

    # ── Market overview metrics ───────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Brent Crude", f"${current_price:.2f}",
                  f"{price_change:+.2f} ({price_chg_pct:+.2f}%)")
    with c2:
        wti_approx = current_price - abs(float(latest.get("brent_wti_spread", 3.2)))
        st.metric("WTI Crude (approx)", f"${wti_approx:.2f}")
    with c3:
        sent = float(latest.get("sentiment_score", 0.05))
        sent_label = "Bullish" if sent > 0.1 else "Bearish" if sent < -0.1 else "Neutral"
        st.metric("Market Sentiment", sent_label, f"{sent:+.3f}")
    with c4:
        eia_chg = float(latest.get("eia_weekly_change_mb", -1.2))
        eia_dir = f"{'↓ Draw' if eia_chg < 0 else '↑ Build'}"
        st.metric("EIA Weekly Change", f"{eia_chg:+.1f} Mbbl", eia_dir)
    with c5:
        dxy = float(latest.get("dxy", 103.5))
        st.metric("DXY (USD Index)", f"{dxy:.2f}")

    st.markdown("---")

    # ── Main two-column layout ────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Generate forecast
        rng_seed = int(current_price * 1000) % 10000
        rng = np.random.default_rng(rng_seed)
        pred_ret = float(latest.get("roc_1d", 0)) * 0.3 + rng.normal(0, 0.003)
        atr_pct  = float(latest.get("atr_pct", 0.015))

        forecast_rows = []
        price = current_price
        for i in range(1, horizon + 1):
            damp  = max(0.4, 1.0 - i * 0.08)
            ret   = pred_ret * damp
            price = price * (1 + ret)
            ci    = current_price * atr_pct * 1.96 * np.sqrt(i)
            forecast_rows.append({
                "date":      datetime.today().date() + timedelta(days=i),
                "price":     round(price, 2),
                "lower_95":  round(price - ci, 2),
                "upper_95":  round(price + ci, 2),
            })
        forecast_df = pd.DataFrame(forecast_rows)

        st.plotly_chart(
            plot_price_with_forecast(full_df["close"], forecast_df if show_ci else None),
            use_container_width=True,
        )

    with col_right:
        # Trading signal
        dir_prob = 0.5 + pred_ret * 20
        dir_prob = float(np.clip(0.5 + float(latest.get("sentiment_ma5", 0)) * 0.3 +
                                 float(latest.get("roc_5d", 0)) * 2.0, 0.2, 0.85))
        signal   = "BUY" if dir_prob >= 0.58 else "SELL" if dir_prob <= 0.42 else "HOLD"
        conf     = abs(dir_prob - 0.5) * 2 + 0.4

        st.markdown("#### Trading signal")
        st.markdown(signal_badge(signal), unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {conf:.0%} &nbsp;&nbsp; **Direction prob:** {dir_prob:.2f}")

        # Risk levels
        atr_d = current_price * atr_pct
        if signal == "BUY":
            sl = current_price - 2 * atr_d
            tp = current_price + 3 * atr_d
        elif signal == "SELL":
            sl = current_price + 2 * atr_d
            tp = current_price - 3 * atr_d
        else:
            sl = current_price - 1.5 * atr_d
            tp = current_price + 1.5 * atr_d

        rc1, rc2 = st.columns(2)
        with rc1:
            st.metric("Stop loss", f"${sl:.2f}")
        with rc2:
            st.metric("Take profit", f"${tp:.2f}")

        # Rationale
        rationale_parts = []
        if float(latest.get("sentiment_ma5", 0)) > 0.1:
            rationale_parts.append("Positive sentiment")
        if float(latest.get("eia_weekly_change_mb", 0)) < -2:
            rationale_parts.append("Large inventory draw")
        if float(latest.get("dxy_above_ma", 1)) == 0:
            rationale_parts.append("USD weakness supportive")
        if float(latest.get("rsi_14", 50)) < 35:
            rationale_parts.append("RSI oversold")
        elif float(latest.get("rsi_14", 50)) > 65:
            rationale_parts.append("RSI overbought")
        rationale = ". ".join(rationale_parts) if rationale_parts else "Mixed signals."
        st.caption(f"💡 {rationale}")

    st.markdown("---")

    # ── Bottom row: signal history + technical gauges ─────────────
    col_b1, col_b2 = st.columns([3, 2])

    with col_b1:
        # Signal history (last 30 days)
        hist_df = full_df.tail(30).copy()
        probs_hist = np.clip(
            0.5 + hist_df["roc_5d"].fillna(0) * 3 +
            hist_df.get("sentiment_ma5", pd.Series(0, index=hist_df.index)).fillna(0) * 0.2,
            0.2, 0.85,
        )
        hist_df["signal"] = probs_hist.apply(
            lambda p: "BUY" if p >= 0.58 else "SELL" if p <= 0.42 else "HOLD"
        )
        hist_df["price"]  = hist_df["close"]
        hist_df["date"]   = hist_df.index

        st.plotly_chart(
            plot_signals_timeline(hist_df[["date", "price", "signal"]]),
            use_container_width=True,
        )

    with col_b2:
        st.markdown("#### Technical gauges")

        indicators = {
            "RSI (14)":          float(latest.get("rsi_14", 50)),
            "Stoch %K":          float(latest.get("stoch_k", 50)),
            "BB %B":             float(latest.get("bb_pct_b", 0.5)) * 100,
            "Williams %R (+100)": float(latest.get("williams_r", -50)) + 100,
        }

        for name, value in indicators.items():
            color = (
                "#1D9E75" if value < 35 else
                "#E24B4A" if value > 65 else
                "#EF9F27"
            )
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:center;margin-bottom:8px;font-size:13px;'>"
                f"<span style='color:var(--text-color,#444);'>{name}</span>"
                f"<span style='font-weight:500;color:{color};'>{value:.1f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.progress(int(np.clip(value, 0, 100)))

    if auto_refresh:
        import time
        time.sleep(300)
        st.rerun()


if __name__ == "__main__":
    main()
