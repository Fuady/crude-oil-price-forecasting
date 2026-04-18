"""
src/serving/schemas.py
-----------------------
Pydantic v2 request/response models for the forecast API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import date


class CrudeType(str, Enum):
    BRENT = "brent"
    WTI   = "wti"


class ForecastRequest(BaseModel):
    crude_type:         CrudeType = Field(default=CrudeType.BRENT)
    horizon_days:       int = Field(default=7, ge=1, le=30,
                                    description="Forecast horizon in trading days")
    include_sentiment:  bool = Field(default=True)
    include_ci:         bool = Field(default=True,
                                     description="Include 95% confidence intervals")


class PricePoint(BaseModel):
    date:       str
    price:      float
    lower_95:   Optional[float] = None
    upper_95:   Optional[float] = None
    return_pct: Optional[float] = None


class KeyDriver(BaseModel):
    factor:    str
    impact:    str          # "bullish" or "bearish"
    magnitude: float


class TradingSignal(BaseModel):
    signal:           str   # BUY | SELL | HOLD
    confidence:       float
    signal_strength:  str   # STRONG | MODERATE | WEAK
    rationale:        str
    stop_loss:        float
    take_profit:      float
    position_size_pct: float


class ForecastResponse(BaseModel):
    crude_type:       str
    current_price:    float
    current_date:     str
    forecasts:        List[PricePoint]
    trading_signal:   TradingSignal
    key_drivers:      List[KeyDriver]
    model_version:    str
    inference_time_ms: float
    disclaimer:       str = (
        "For educational purposes only. Not financial advice. "
        "Past performance does not guarantee future results."
    )


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str
    last_data_date: Optional[str]
    uptime_seconds: float


class MarketContextResponse(BaseModel):
    crude_type:     str
    current_price:  float
    price_change_1d: float
    price_change_5d: float
    rsi_14:         Optional[float]
    sentiment_score: Optional[float]
    eia_last_change_mb: Optional[float]
    vol_regime:     str   # "HIGH" or "NORMAL"
    date:           str
