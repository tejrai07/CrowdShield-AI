"""Tests for FastAPI backend routes using Starlette TestClient."""
from __future__ import annotations

from starlette.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_health_check() -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "service": "crowdshield-backend"}


def test_dashboard_summary() -> None:
    res = client.get("/api/dashboard/summary")
    assert res.status_code == 200
    data = res.json()
    assert "crowdStatus" in data
    assert "activePeopleCount" in data
    assert "currentRiskLevel" in data


def test_zones_endpoint() -> None:
    res = client.get("/api/zones")
    assert res.status_code == 200
    zones = res.json()
    assert isinstance(zones, list)
    assert len(zones) > 0
    assert "id" in zones[0]
    assert "occupancy" in zones[0]
    assert "riskLevel" in zones[0]


def test_risk_forecast_endpoint() -> None:
    res = client.get("/api/risk/forecast")
    assert res.status_code == 200
    forecast = res.json()
    assert len(forecast) == 3
    horizons = [f["horizon"] for f in forecast]
    assert horizons == ["CURRENT", "+5min", "+10min"]


def test_pipeline_status_endpoint() -> None:
    res = client.get("/api/pipeline/status")
    assert res.status_code == 200
    data = res.json()
    assert "pipeline_status" in data
    assert "model_data_notice" in data
    assert len(data["model_data_notice"]) > 0


def test_pipeline_worker_endpoints() -> None:
    # Worker status
    res = client.get("/api/pipeline/worker")
    assert res.status_code == 200
    assert "status" in res.json()
    assert "is_running" in res.json()

    # Worker stop
    res = client.post("/api/pipeline/stop")
    assert res.status_code == 200
    assert res.json()["is_running"] is False

