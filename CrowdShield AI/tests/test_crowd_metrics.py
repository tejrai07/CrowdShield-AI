"""Unit tests for features/crowd_metrics.py.

All synthetic — no YOLO, no ByteTrack, no video. `Track`-like objects are
SimpleNamespace instances exposing only `.track_id` and `.center`, since
that's all CrowdMetricsCalculator actually reads.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from config.schema import ZoneConfig
from features.crowd_metrics import (
    CrowdMetricsCalculator,
    circular_mean_and_variance,
    track_direction_deg,
    track_speed_px_s,
)
from features.zones import ZoneManager


def _track(track_id: int, cx: float, cy: float):
    return SimpleNamespace(track_id=track_id, center=(cx, cy))


def _zone(id_, area_m2=10.0, enabled=True, polygon=None):
    polygon = polygon or [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    return ZoneConfig(id=id_, name=id_.title(), polygon=polygon, area_m2=area_m2, enabled=enabled)


# --- people count / density -------------------------------------------------

def test_people_count_matches_tracks_in_zone():
    zones = [_zone("zone_a")]
    calc = CrowdMetricsCalculator(ZoneManager(zones))

    tracks = [_track(1, 10, 10), _track(2, 20, 20), _track(3, 500, 500)]  # 3rd is outside
    trajectories = {1: [(0.0, 10.0, 10.0)], 2: [(0.0, 20.0, 20.0)], 3: [(0.0, 500.0, 500.0)]}

    snapshot = calc.compute_snapshot(tracks, trajectories, timestamp_s=0.0)
    assert snapshot["zone_a"].people_count == 2


def test_density_formula():
    zones = [_zone("zone_a", area_m2=4.0)]
    calc = CrowdMetricsCalculator(ZoneManager(zones))

    tracks = [_track(1, 10, 10), _track(2, 20, 20)]
    trajectories = {1: [(0.0, 10.0, 10.0)], 2: [(0.0, 20.0, 20.0)]}

    snapshot = calc.compute_snapshot(tracks, trajectories, timestamp_s=0.0)
    assert snapshot["zone_a"].density == pytest.approx(2 / 4.0)


# --- speed -------------------------------------------------------------------

def test_track_speed_known_displacement_and_time():
    # displacement (3,4) over 1s -> distance 5 -> speed 5 px/s
    trajectory = [(0.0, 0.0, 0.0), (1.0, 3.0, 4.0)]
    assert track_speed_px_s(trajectory) == pytest.approx(5.0)


def test_track_speed_none_with_insufficient_history():
    assert track_speed_px_s([(0.0, 0.0, 0.0)]) is None
    assert track_speed_px_s([]) is None


def test_track_speed_stationary_approaches_zero():
    trajectory = [(0.0, 50.0, 50.0), (1.0, 50.0, 50.0)]
    assert track_speed_px_s(trajectory) == pytest.approx(0.0)


def test_avg_speed_across_zone():
    zones = [_zone("zone_a")]
    calc = CrowdMetricsCalculator(ZoneManager(zones))

    tracks = [_track(1, 10, 10), _track(2, 20, 20)]
    trajectories = {
        1: [(0.0, 0.0, 0.0), (1.0, 10.0, 0.0)],   # speed 10
        2: [(0.0, 0.0, 0.0), (1.0, 20.0, 0.0)],   # speed 20
    }
    snapshot = calc.compute_snapshot(tracks, trajectories, timestamp_s=1.0)
    assert snapshot["zone_a"].avg_speed_px_s == pytest.approx(15.0)


def test_speed_variance_known_values():
    zones = [_zone("zone_a")]
    calc = CrowdMetricsCalculator(ZoneManager(zones))

    tracks = [_track(1, 10, 10), _track(2, 20, 20), _track(3, 30, 30)]
    trajectories = {
        1: [(0.0, 0.0, 0.0), (1.0, 10.0, 0.0)],   # speed 10
        2: [(0.0, 0.0, 0.0), (1.0, 20.0, 0.0)],   # speed 20
        3: [(0.0, 0.0, 0.0), (1.0, 30.0, 0.0)],   # speed 30
    }
    snapshot = calc.compute_snapshot(tracks, trajectories, timestamp_s=1.0)
    # mean=20, population variance = ((10-20)^2+(0)+(10)^2)/3 = (100+0+100)/3
    assert snapshot["zone_a"].speed_variance == pytest.approx(200 / 3)


def test_speed_variance_single_observation_is_zero_by_convention():
    zones = [_zone("zone_a")]
    calc = CrowdMetricsCalculator(ZoneManager(zones))

    tracks = [_track(1, 10, 10)]
    trajectories = {1: [(0.0, 0.0, 0.0), (1.0, 10.0, 0.0)]}
    snapshot = calc.compute_snapshot(tracks, trajectories, timestamp_s=1.0)
    assert snapshot["zone_a"].avg_speed_px_s == pytest.approx(10.0)
    assert snapshot["zone_a"].speed_variance == 0.0


def test_speed_none_when_zero_valid_observations():
    zones = [_zone("zone_a")]
    calc = CrowdMetricsCalculator(ZoneManager(zones))

    # Track present but with only 1 trajectory point -> no speed possible.
    tracks = [_track(1, 10, 10)]
    trajectories = {1: [(0.0, 10.0, 10.0)]}
    snapshot = calc.compute_snapshot(tracks, trajectories, timestamp_s=0.0)
    assert snapshot["zone_a"].avg_speed_px_s is None
    assert snapshot["zone_a"].speed_variance is None


def test_speed_never_mislabeled_as_mps():
    """Regression/documentation guard: the field name itself must remain
    avg_speed_px_s, not avg_speed_mps or similar — this test fails loudly
    if someone renames the field to imply calibrated units."""
    zones = [_zone("zone_a")]
    calc = CrowdMetricsCalculator(ZoneManager(zones))
    tracks = [_track(1, 10, 10)]
    trajectories = {1: [(0.0, 0.0, 0.0), (1.0, 10.0, 0.0)]}
    snapshot = calc.compute_snapshot(tracks, trajectories, timestamp_s=1.0)

    fields = snapshot["zone_a"].__dataclass_fields__
    assert "avg_speed_px_s" in fields
    assert "avg_speed_mps" not in fields
    assert not any("mps" in f or "m_s" in f or "meters" in f for f in fields)


# --- direction -----------------------------------------------------------------

def test_direction_moving_right_is_zero_degrees():
    trajectory = [(0.0, 0.0, 0.0), (1.0, 10.0, 0.0)]
    assert track_direction_deg(trajectory) == pytest.approx(0.0)


def test_direction_moving_down_is_90_degrees():
    # image coords: y increases downward
    trajectory = [(0.0, 0.0, 0.0), (1.0, 0.0, 10.0)]
    assert track_direction_deg(trajectory) == pytest.approx(90.0)


def test_direction_moving_left_is_180_degrees():
    trajectory = [(0.0, 10.0, 0.0), (1.0, 0.0, 0.0)]
    assert track_direction_deg(trajectory) == pytest.approx(180.0)


def test_direction_moving_up_is_270_degrees():
    trajectory = [(0.0, 0.0, 10.0), (1.0, 0.0, 0.0)]
    assert track_direction_deg(trajectory) == pytest.approx(270.0)


def test_direction_none_for_stationary_track():
    trajectory = [(0.0, 50.0, 50.0), (1.0, 50.0, 50.0)]
    assert track_direction_deg(trajectory) is None


def test_direction_none_with_insufficient_history():
    assert track_direction_deg([(0.0, 0.0, 0.0)]) is None


# --- circular direction / dispersion --------------------------------------------

def test_circular_wraparound_near_zero_not_180():
    mean, variance = circular_mean_and_variance([359.0, 1.0])
    # Must be near 0 (or equivalently 360), NOT the naive arithmetic
    # average of 180.
    assert mean < 1.0 or mean > 359.0
    assert variance < 0.01  # highly aligned


def test_circular_aligned_tracks_low_variance():
    mean, variance = circular_mean_and_variance([10.0, 12.0, 8.0])
    assert variance < 0.01
    assert 8.0 <= mean <= 12.0


def test_circular_opposing_tracks_high_variance():
    _, variance = circular_mean_and_variance([0.0, 180.0])
    assert variance == pytest.approx(1.0, abs=1e-6)


def test_circular_dispersed_tracks_high_variance():
    _, variance = circular_mean_and_variance([0.0, 90.0, 180.0, 270.0])
    assert variance == pytest.approx(1.0, abs=1e-6)


def test_circular_single_angle_is_trivially_aligned():
    mean, variance = circular_mean_and_variance([45.0])
    assert mean == pytest.approx(45.0)
    assert variance == 0.0


def test_circular_empty_returns_none():
    mean, variance = circular_mean_and_variance([])
    assert mean is None
    assert variance is None


def test_flow_direction_none_when_no_valid_directions_in_zone():
    zones = [_zone("zone_a")]
    calc = CrowdMetricsCalculator(ZoneManager(zones))
    tracks = [_track(1, 10, 10)]
    trajectories = {1: [(0.0, 10.0, 10.0)]}  # insufficient history
    snapshot = calc.compute_snapshot(tracks, trajectories, timestamp_s=0.0)
    assert snapshot["zone_a"].flow_direction_deg is None
    assert snapshot["zone_a"].direction_variance is None


# --- inflow / outflow ---------------------------------------------------------

def test_inflow_when_entering_from_outside():
    zones = [_zone("zone_a")]
    calc = CrowdMetricsCalculator(ZoneManager(zones))

    tracks = [_track(1, 50, 50)]
    trajectories = {1: [(0.0, 50.0, 50.0)]}
    snapshot = calc.compute_snapshot(tracks, trajectories, timestamp_s=0.0)
    assert snapshot["zone_a"].inflow_count == 1
    assert snapshot["zone_a"].outflow_count == 0


def test_outflow_when_leaving_to_outside():
    zones = [_zone("zone_a")]
    manager = ZoneManager(zones)
    calc = CrowdMetricsCalculator(manager)

    # frame 1: enters zone_a
    calc.compute_snapshot([_track(1, 50, 50)], {1: [(0.0, 50.0, 50.0)]}, timestamp_s=0.0)
    # frame 2: moves outside all zones
    snapshot = calc.compute_snapshot([_track(1, 500, 500)], {1: [(1.0, 500.0, 500.0)]}, timestamp_s=1.0)
    assert snapshot["zone_a"].outflow_count == 1
    assert snapshot["zone_a"].inflow_count == 0


def test_zone_to_zone_transition_increments_both():
    zone_a = _zone("zone_a", polygon=[(0, 0), (50, 0), (50, 100), (0, 100)])
    zone_b = _zone("zone_b", polygon=[(60, 0), (110, 0), (110, 100), (60, 100)])
    manager = ZoneManager([zone_a, zone_b])
    calc = CrowdMetricsCalculator(manager)

    # frame 1: in zone_a
    calc.compute_snapshot([_track(1, 25, 50)], {1: [(0.0, 25.0, 50.0)]}, timestamp_s=0.0)
    # frame 2: in zone_b
    snapshot = calc.compute_snapshot([_track(1, 85, 50)], {1: [(1.0, 85.0, 50.0)]}, timestamp_s=1.0)

    assert snapshot["zone_a"].outflow_count == 1
    assert snapshot["zone_b"].inflow_count == 1


def test_no_duplicate_transitions_while_remaining_in_zone():
    zones = [_zone("zone_a")]
    manager = ZoneManager(zones)
    calc = CrowdMetricsCalculator(manager)

    inflow_total = 0
    for i in range(20):
        snapshot = calc.compute_snapshot(
            [_track(1, 50, 50)], {1: [(float(i), 50.0, 50.0)]}, timestamp_s=float(i)
        )
        inflow_total += snapshot["zone_a"].inflow_count

    assert inflow_total == 1, "a person staying in zone_a for 20 frames must generate exactly 1 inflow event"


# --- empty / disabled zones -----------------------------------------------------

def test_empty_zone_defaults():
    zones = [_zone("zone_a")]
    calc = CrowdMetricsCalculator(ZoneManager(zones))

    snapshot = calc.compute_snapshot([], {}, timestamp_s=0.0)
    metrics = snapshot["zone_a"]
    assert metrics.people_count == 0
    assert metrics.density == 0.0
    assert metrics.avg_speed_px_s is None
    assert metrics.speed_variance is None
    assert metrics.flow_direction_deg is None
    assert metrics.direction_variance is None
    assert metrics.inflow_count == 0
    assert metrics.outflow_count == 0


def test_disabled_zone_produces_no_metrics_entry():
    zones = [_zone("zone_a", enabled=False)]
    calc = CrowdMetricsCalculator(ZoneManager(zones))

    snapshot = calc.compute_snapshot([], {}, timestamp_s=0.0)
    assert "zone_a" not in snapshot


def test_disabled_zone_ignored_even_with_tracks_geometrically_inside():
    zones = [_zone("zone_a", enabled=False)]
    calc = CrowdMetricsCalculator(ZoneManager(zones))

    tracks = [_track(1, 50, 50)]
    trajectories = {1: [(0.0, 50.0, 50.0)]}
    snapshot = calc.compute_snapshot(tracks, trajectories, timestamp_s=0.0)
    assert snapshot == {}
