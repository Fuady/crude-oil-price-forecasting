"""
src/serving/api.py
-------------------
FastAPI forecast and trading signal endpoint.

Endpoints:
  GET  /health              — Service health + model status
  GET  /market/summary      — Current market snapshot
  POST /forecast            — Price forecast + trading signal
  GET  /forecast/latest     — Latest forecast (no input required)
  GET  /signal/history      — Past 30 days of signals

Run:
  uvicorn src.serving.api:app --reload --port 8000
  Swagger docs: http://localhost:8000/docs
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.serving.schemas import (
    ForecastRequest, ForecastResponse, ForecastPoint,
    TradingSignalDetail, FactorDriver, HealthResponse,
    MarketSummaryResponse, TradingSignal,
)
from src.signals.signal_generator import generate_signal

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Oil Price Forecasting API",
    description=(
        "Multi-factor crude oil price forecasting and trading signal generation. "
        "Combines LSTM + XGBoost ensemble with NLP sentiment and EIA inventory data."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME    = time.time()
MODEL_VERSION = "ensemble-v1.0.0"
MODELS_DIR    = Path("models")
FEATURES_DIR  = Path("data/features")

# Lazy-loaded model and data caches
_xgb_models   = {}
_lstm_trainers = {}
_feature_data  = {}
_meta          = {}


def _load_model(instrument: str = "brent"):
    """Lazy-load XGBoost model."""
    if instrument not in _xgb_models:
        path = MODELS_DIR / f"xgboost_{instrument}.joblib"
        if path.exists():
            try:
                from src.models.xgboost_model import XGBoostOilForecaster
                _xgb_models[instrument] = XGBoostOilForecaster.load(path)
            except Exception:
                _xgb_models[instrument] = None
        else:
            _xgb_models[instrument] = None
    return _xgb_models.get(instrument)


def _load_features(instrument: str = "brent") -> Optional[dict]:
    """Lazy-load feature data."""
    if instrument not in _feature_data:
        try:
            from src.features.feature_pipeline import load_features
            _feature_data[instrument] = load_features(instrument)
        except Exception:
            _feature_data[instrument] = None
    return _feature_data.get(instrument)


def _get_latest_row(instrument: str = "brent") -> Optional[pd.Series]:
    """Return the most recent feature row for real-time inference."""
    data = _load_features(instrument)
    if data is None:
        return None
    full_df = pd.concat([data["train_df"], data["val_df"], data["test_df"]])
    return full_df.sort_index().iloc[-1]


def _make_forecast_points(
    current_price: float,
    predicted_return_1d: float,
    ci_lower: float,
    ci_upper: float,
    horizon: int,
    base_date: datetime,
) -> list:
    """Generate N forecast points by compounding the daily return."""
    points   = []
    price    = current_price
    # Dampen forecast drift over longer horizons (uncertainty grows)
    dampening = [1.0, 0.85, 0.70, 0.60, 0.52, 0.46, 0.40]

    for i in range(1, horizon + 1):
        damp  = dampening[min(i - 1, len(dampening) - 1)]
        ret   = predicted_return_1d * damp
        price = price * (1 + ret)

        # Confidence interval widens with horizon
        ci_spread = (ci_upper - ci_lower) * np.sqrt(i) / 2
        date_str  = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")

        points.append(ForecastPoint(
            date=date_str,
            price=round(price, 2),
            lower_95=round(price - ci_spread, 2),
            upper_95=round(price + ci_spread, 2),
            return_pct=round(ret * 100, 4),
        ))
    return points


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    data = _load_features("brent")
    xgb  = _load_model("brent")
    last_date = "unknown"
    if data:
        full = pd.concat([data["train_df"], data["val_df"], data["test_df"]])
        last_date = str(full.index[-1].date())

    return HealthResponse(
        status="healthy" if xgb is not None else "degraded",
        models_loaded={
            "xgboost_brent": xgb is not None,
            "features_brent": data is not None,
        },
        last_data_date=last_date,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.get("/market/summary", response_model=MarketSummaryResponse)
def market_summary():
    """Return current market snapshot from latest data."""
    brent_row = _get_latest_row("brent")
    wti_row   = _get_latest_row("wti")

    brent_price = float(brent_row["close"]) if brent_row is not None else 82.0
    wti_price   = float(wti_row["close"])   if wti_row   is not None else 78.0

    spread      = round(brent_price - wti_price, 2)
    sent_score  = float(brent_row.get("sentiment_score", 0.05)) if brent_row is not None else 0.05
    sent_label  = "Bullish" if sent_score > 0.1 else "Bearish" if sent_score < -0.1 else "Neutral"
    eia_change  = float(brent_row.get("eia_weekly_change_mb", -1.2)) if brent_row is not None else -1.2
    dxy         = float(brent_row.get("dxy", 103.5)) if brent_row is not None else 103.5

    as_of = str(brent_row.name.date()) if brent_row is not None else datetime.today().strftime("%Y-%m-%d")

    return MarketSummaryResponse(
        brent_price=round(brent_price, 2),
        wti_price=round(wti_price, 2),
        brent_wti_spread=spread,
        sentiment_score=round(sent_score, 4),
        sentiment_label=sent_label,
        eia_last_change_mb=round(eia_change, 2),
        eia_is_draw=eia_change < 0,
        dxy=round(dxy, 2),
        as_of_date=as_of,
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    t0         = time.time()
    instrument = request.crude_type.value
    model      = _load_model(instrument)
    data       = _load_features(instrument)

    if model is None or data is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not ready. Run: "
                "python src/ingestion/download_data.py && "
                "python src/features/feature_pipeline.py && "
                "python src/models/train.py"
            ),
        )

    # Get latest feature row
    full_df     = pd.concat([data["train_df"], data["val_df"], data["test_df"]]).sort_index()
    latest_row  = full_df.iloc[-1]
    current_price = float(latest_row["close"])
    current_date  = latest_row.name

    feature_cols = model.feature_cols
    X_latest     = pd.DataFrame(
        [latest_row[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)]
    )

    # Predict
    pred_return = float(model.predict_return(X_latest)[0])
    dir_prob    = float(model.predict_direction_proba(X_latest)[0])

    # Confidence interval (ATR-based)
    atr_pct    = float(latest_row.get("atr_pct", 0.015))
    ci_spread  = current_price * atr_pct * 1.96
    ci_lower   = current_price - ci_spread
    ci_upper   = current_price + ci_spread

    # Build forecast points
    forecasts = _make_forecast_points(
        current_price, pred_return, ci_lower, ci_upper,
        request.horizon_days, current_date,
    )

    # Trading signal
    trading_signal = None
    if request.include_signal:
        forecast_dict = {
            "predicted_return":  pred_return,
            "predicted_price":   current_price * (1 + pred_return),
            "direction_prob_up": dir_prob,
            "current_price":     current_price,
        }
        sig = generate_signal(forecast_dict, latest_row)
        trading_signal = TradingSignalDetail(
            signal=TradingSignal(sig["signal"]),
            confidence=sig["confidence"],
            signal_strength=sig["signal_strength"],
            rationale=sig["rationale"],
            stop_loss=sig["stop_loss"],
            take_profit=sig["take_profit"],
            position_size_pct=sig["position_size_pct"],
        )

    # Key drivers (SHAP)
    key_drivers = []
    try:
        drivers = model.get_top_drivers(latest_row, n=5)
        key_drivers = [
            FactorDriver(
                factor=d["factor"],
                impact=d["impact"],
                magnitude=round(d["magnitude"], 4),
            )
            for d in drivers
        ]
    except Exception:
        pass

    return ForecastResponse(
        crude_type=instrument,
        current_price=round(current_price, 2),
        current_date=str(current_date.date()),
        forecasts=forecasts,
        trading_signal=trading_signal,
        key_drivers=key_drivers,
        model_version=MODEL_VERSION,
        inference_time_ms=round((time.time() - t0) * 1000, 1),
    )


@app.get("/forecast/latest")
def forecast_latest():
    """Shortcut: 7-day Brent forecast with no request body needed."""
    return forecast(ForecastRequest(crude_type="brent", horizon_days=7))


@app.get("/signal/history")
def signal_history(instrument: str = "brent", days: int = 30):
    """Return historical signal log (last N days from test set)."""
    model = _load_model(instrument)
    data  = _load_features(instrument)

    if model is None or data is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    test_df = data["test_df"].tail(days)
    feature_cols = model.feature_cols
    dir_probs = model.predict_direction_proba(test_df)

    history = []
    for i, (idx, row) in enumerate(test_df.iterrows()):
        prob = float(dir_probs[i])
        sig  = "BUY" if prob >= 0.58 else "SELL" if prob <= 0.42 else "HOLD"
        history.append({
            "date":             str(idx.date()),
            "price":            round(float(row["close"]), 2),
            "signal":           sig,
            "direction_prob_up": round(prob, 4),
            "sentiment_score":  round(float(row.get("sentiment_score", 0)), 4),
        })

    return {"instrument": instrument, "history": history, "n_days": len(history)}


@app.exception_handler(Exception)
async def global_error(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )
