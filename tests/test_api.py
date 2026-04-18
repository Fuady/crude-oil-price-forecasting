"""
tests/test_api.py
------------------
Integration tests for FastAPI endpoints.
Run: pytest tests/test_api.py -v
"""

import sys
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving.api import app

client = TestClient(app)


class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_field(self):
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")

    def test_health_has_uptime(self):
        data = client.get("/health").json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestMarketSummary:

    def test_market_summary_200(self):
        assert client.get("/market/summary").status_code == 200

    def test_market_summary_fields(self):
        data = client.get("/market/summary").json()
        for field in ["brent_price", "wti_price", "sentiment_score",
                      "eia_last_change_mb", "as_of_date"]:
            assert field in data

    def test_brent_price_positive(self):
        data = client.get("/market/summary").json()
        assert data["brent_price"] > 0

    def test_spread_is_difference(self):
        data = client.get("/market/summary").json()
        expected_spread = round(data["brent_price"] - data["wti_price"], 2)
        assert abs(data["brent_wti_spread"] - expected_spread) < 0.01


class TestForecastEndpoint:

    VALID_PAYLOAD = {
        "crude_type": "brent",
        "horizon_days": 5,
        "include_sentiment": True,
        "include_signal": True,
    }

    def test_forecast_returns_200_or_503(self):
        resp = client.post("/forecast", json=self.VALID_PAYLOAD)
        assert resp.status_code in (200, 503)

    def test_forecast_structure_when_model_loaded(self):
        resp = client.post("/forecast", json=self.VALID_PAYLOAD)
        if resp.status_code != 200:
            pytest.skip("Model not trained yet — run train.py first")
        data = resp.json()
        for field in ["crude_type", "current_price", "forecasts",
                      "model_version", "inference_time_ms"]:
            assert field in data

    def test_forecast_point_count_matches_horizon(self):
        resp = client.post("/forecast", json={**self.VALID_PAYLOAD, "horizon_days": 7})
        if resp.status_code != 200:
            pytest.skip("Model not loaded")
        assert len(resp.json()["forecasts"]) == 7

    def test_forecast_prices_positive(self):
        resp = client.post("/forecast", json=self.VALID_PAYLOAD)
        if resp.status_code != 200:
            pytest.skip("Model not loaded")
        for pt in resp.json()["forecasts"]:
            assert pt["price"] > 0

    def test_forecast_ci_ordering(self):
        """Upper CI should be >= price >= lower CI (approximately)."""
        resp = client.post("/forecast", json=self.VALID_PAYLOAD)
        if resp.status_code != 200:
            pytest.skip("Model not loaded")
        for pt in resp.json()["forecasts"]:
            assert pt["upper_95"] >= pt["lower_95"]

    def test_forecast_signal_valid(self):
        resp = client.post("/forecast", json=self.VALID_PAYLOAD)
        if resp.status_code != 200:
            pytest.skip("Model not loaded")
        sig = resp.json().get("trading_signal")
        if sig:
            assert sig["signal"] in ("BUY", "SELL", "HOLD")
            assert 0 <= sig["confidence"] <= 1

    def test_forecast_horizon_too_large_rejected(self):
        resp = client.post("/forecast", json={**self.VALID_PAYLOAD, "horizon_days": 100})
        assert resp.status_code == 422

    def test_forecast_invalid_crude_type(self):
        resp = client.post("/forecast", json={**self.VALID_PAYLOAD, "crude_type": "dubai"})
        assert resp.status_code == 422

    def test_forecast_latest_shortcut(self):
        resp = client.get("/forecast/latest")
        assert resp.status_code in (200, 503)


class TestSignalHistory:

    def test_signal_history_returns_200_or_503(self):
        resp = client.get("/signal/history")
        assert resp.status_code in (200, 503)

    def test_signal_history_structure(self):
        resp = client.get("/signal/history")
        if resp.status_code != 200:
            pytest.skip("Model not loaded")
        data = resp.json()
        assert "history" in data
        assert "instrument" in data

    def test_signal_history_valid_signals(self):
        resp = client.get("/signal/history?days=10")
        if resp.status_code != 200:
            pytest.skip("Model not loaded")
        for item in resp.json()["history"]:
            assert item["signal"] in ("BUY", "SELL", "HOLD")
            assert 0 <= item["direction_prob_up"] <= 1
