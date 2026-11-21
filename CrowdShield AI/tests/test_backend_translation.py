"""Tests for backend ML-to-UI translation logic."""
from __future__ import annotations

from backend.translation import (
    generate_alerts_and_events,
    map_zone_status,
    translate_bottleneck_pressure,
    translate_dashboard_summary,
    translate_feature_contributions,
    translate_flow_directions,
    translate_forecast,
    translate_zone,
    translate_zones,
)


def test_map_zone_status() -> None:
    assert map_zone_status("disabled", True, "SAFE") == "OFFLINE"
    assert map_zone_status("unavailable", True, "SAFE") == "OFFLINE"
    assert map_zone_status("no_data", True, "SAFE") == "OFFLINE"
    assert map_zone_status("ok", False, "SAFE") == "OFFLINE"
    assert map_zone_status("ok", True, "CRITICAL") == "RESTRICTED"
    assert map_zone_status("ok", True, "HIGH") == "RESTRICTED"
    assert map_zone_status("ok", True, "MODERATE") == "ELEVATED"
    assert map_zone_status("ok", True, "LOW") == "NORMAL"
    assert map_zone_status("ok", True, "SAFE") == "NORMAL"


def test_translate_zone_occupancy_and_density() -> None:
    ml_zone = {
        "zone_id": "zone_1",
        "name": "Gate A",
        "capacity": 200,
        "utilization": 0.456,
        "density": 1.234,
        "status": "ok",
        "enabled": True,
        "risk": {"current": {"risk_class": "MODERATE"}},
    }
    z = translate_zone(ml_zone, "2026-11-21T18:00:00Z")
    assert z.id == "zone_1"
    assert z.name == "Gate A"
    assert z.capacity == 200
    assert z.occupancy == 45.6
    assert z.density == 1.23
    assert z.status == "ELEVATED"
    assert z.riskLevel == "MODERATE"
    assert z.updatedAt == "2026-11-21T18:00:00Z"


def test_translate_flow_directions_image_coordinates() -> None:
    # Image coordinates: 90° = South, 270° = North, 0° = East, 180° = West
    frame = {
        "zones": [
            {"metrics": {"flow_direction_deg": 90.0}, "people_count": 10},
            {"metrics": {"flow_direction_deg": 270.0}, "people_count": 20},
            {"metrics": {"flow_direction_deg": 0.0}, "people_count": 5},
            {"metrics": {"flow_direction_deg": 180.0}, "people_count": 15},
        ]
    }
    slices = translate_flow_directions(frame)
    by_dir = {s.direction: s.percentage for s in slices}
    assert by_dir["SOUTH"] == 20.0  # 10 / 50 = 20%
    assert by_dir["NORTH"] == 40.0  # 20 / 50 = 40%
    assert by_dir["EAST"] == 10.0   # 5 / 50 = 10%
    assert by_dir["WEST"] == 30.0   # 15 / 50 = 30%


def test_generate_alerts_for_unsupported_and_incomplete_zones() -> None:
    frame = {
        "zones": [
            {
                "zone_id": "zone_crit",
                "name": "Chokepoint 1",
                "risk": {"current": {"risk_class": "CRITICAL", "risk_score": 92.0}},
            }
        ],
        "incomplete_input_zones": ["zone_stationary"],
        "unsupported_input_zones": ["zone_unsupported"],
        "warnings": ["Temporal warmup incomplete"],
        "timestamp_s": 10.0,
    }
    alerts, events = generate_alerts_and_events(frame)

    severities = [a.severity for a in alerts]
    assert "CRITICAL" in severities
    assert "MEDIUM" in severities  # incomplete inputs
    assert "HIGH" in severities    # unsupported out-of-distribution inputs
    assert "LOW" in severities     # warnings

    # Event generated for CRITICAL
    assert len(events) >= 1
    assert events[0].severity == "CRITICAL"
