"""Unit tests for features/zones.py.

All synthetic — no YOLO, no ByteTrack, no video. `Track`-like objects for
assignment tests are simple objects/namedtuples exposing `.center`, since
that's the only attribute ZoneManager.assign_track() actually reads.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from config.schema import ZoneConfig
from features.zones import ZoneManager, ZoneTransitionTracker, point_in_polygon

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

# L-shaped concave polygon: a 10x10 square with a 5x5 notch cut out of the
# top-right corner.
CONCAVE_L = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (5.0, 5.0), (5.0, 10.0), (0.0, 10.0)]


def _zone(id_, name, polygon, area_m2=10.0, enabled=True) -> ZoneConfig:
    return ZoneConfig(id=id_, name=name, polygon=polygon, area_m2=area_m2, enabled=enabled)


def _track(cx: float, cy: float):
    """Minimal stand-in for vision.tracker.Track — only `.center` is used
    by ZoneManager.assign_track()."""
    return SimpleNamespace(center=(cx, cy))


# --- point_in_polygon: simple polygon --------------------------------------

def test_point_inside_simple_polygon():
    assert point_in_polygon(5, 5, SQUARE) is True


def test_point_outside_simple_polygon():
    assert point_in_polygon(15, 5, SQUARE) is False


# --- point_in_polygon: boundary behavior ------------------------------------

def test_point_on_edge_counts_as_inside():
    assert point_in_polygon(0, 5, SQUARE) is True  # left edge, midpoint


def test_point_on_corner_counts_as_inside():
    assert point_in_polygon(0, 0, SQUARE) is True
    assert point_in_polygon(10, 10, SQUARE) is True


def test_point_just_outside_edge_is_outside():
    assert point_in_polygon(-0.01, 5, SQUARE) is False


# --- point_in_polygon: concave polygon --------------------------------------

def test_concave_polygon_notch_is_outside():
    # (7, 7) falls in the notch cut out of the L-shape — must be outside.
    assert point_in_polygon(7, 7, CONCAVE_L) is False


def test_concave_polygon_main_body_is_inside():
    assert point_in_polygon(2, 2, CONCAVE_L) is True


def test_concave_polygon_leg_is_inside():
    # (2, 8) is in the L's vertical leg, still inside despite the notch.
    assert point_in_polygon(2, 8, CONCAVE_L) is True


# --- ZoneManager: multiple zones --------------------------------------------

def test_multiple_zones_point_maps_to_correct_zone():
    zones = [
        _zone("left", "Left", [(0, 0), (10, 0), (10, 10), (0, 10)]),
        _zone("right", "Right", [(20, 0), (30, 0), (30, 10), (20, 10)]),
    ]
    manager = ZoneManager(zones)
    assert manager.assign_point(5, 5) == "left"
    assert manager.assign_point(25, 5) == "right"
    assert manager.assign_point(15, 5) is None  # gap between zones


# --- ZoneManager: overlapping / adjacent zones ------------------------------

def test_overlapping_zones_first_configured_wins():
    zones = [
        _zone("first", "First", [(0, 0), (10, 0), (10, 10), (0, 10)]),
        _zone("second", "Second", [(5, 0), (15, 0), (15, 10), (5, 10)]),  # overlaps "first"
    ]
    manager = ZoneManager(zones)
    # (7, 5) is inside both polygons — "first" is configured first and must win.
    assert manager.assign_point(7, 5) == "first"
    # (12, 5) is only inside "second".
    assert manager.assign_point(12, 5) == "second"


def test_adjacent_zones_shared_edge_first_configured_wins():
    """Two zones sharing a boundary edge (not truly overlapping in area,
    but overlapping on that one edge) must resolve deterministically via
    the same first-configured-wins rule."""
    zones = [
        _zone("left", "Left", [(0, 0), (10, 0), (10, 10), (0, 10)]),
        _zone("right", "Right", [(10, 0), (20, 0), (20, 10), (10, 10)]),
    ]
    manager = ZoneManager(zones)
    assert manager.assign_point(10, 5) == "left"  # shared edge x=10


# --- ZoneManager: disabled zones --------------------------------------------

def test_disabled_zone_is_ignored():
    zones = [
        _zone("disabled_zone", "Disabled", [(0, 0), (10, 0), (10, 10), (0, 10)], enabled=False),
    ]
    manager = ZoneManager(zones)
    assert manager.assign_point(5, 5) is None


def test_disabled_zone_does_not_block_enabled_zone_beneath_it():
    zones = [
        _zone("disabled_zone", "Disabled", [(0, 0), (10, 0), (10, 10), (0, 10)], enabled=False),
        _zone("enabled_zone", "Enabled", [(0, 0), (10, 0), (10, 10), (0, 10)], enabled=True),
    ]
    manager = ZoneManager(zones)
    assert manager.assign_point(5, 5) == "enabled_zone"


# --- ZoneManager: track assignment ------------------------------------------

def test_track_center_correctly_maps_to_zone():
    zones = [_zone("zone_a", "Zone A", SQUARE)]
    manager = ZoneManager(zones)

    track = _track(5.0, 5.0)
    assert manager.assign_track(track) == "zone_a"

    track_outside = _track(50.0, 50.0)
    assert manager.assign_track(track_outside) is None


def test_get_zone_returns_zone_by_id():
    zones = [_zone("zone_a", "Zone A", SQUARE, area_m2=42.0)]
    manager = ZoneManager(zones)
    zone = manager.get_zone("zone_a")
    assert zone is not None
    assert zone.name == "Zone A"
    assert zone.area_m2 == 42.0
    assert manager.get_zone("nonexistent") is None


# --- Config validation -------------------------------------------------------

def test_invalid_polygon_too_few_points_rejected():
    with pytest.raises(Exception):
        ZoneConfig(id="bad", name="Bad", polygon=[(0, 0), (10, 10)], area_m2=10.0)


def test_invalid_polygon_non_numeric_coordinates_rejected():
    with pytest.raises(Exception):
        ZoneConfig(id="bad", name="Bad", polygon=[("a", "b"), (1, 1), (2, 2)], area_m2=10.0)


def test_duplicate_zone_ids_rejected():
    from config.schema import VenueConfig

    with pytest.raises(Exception):
        VenueConfig(
            name="test_venue",
            zones=[
                _zone("dup", "First", SQUARE),
                _zone("dup", "Second", [(20, 0), (30, 0), (30, 10), (20, 10)]),
            ],
        )


def test_negative_area_rejected():
    with pytest.raises(Exception):
        ZoneConfig(id="bad", name="Bad", polygon=SQUARE, area_m2=-5.0)


# --- ZoneTransitionTracker ---------------------------------------------------

def test_zone_transition_tracker_reports_previous_and_current():
    tracker = ZoneTransitionTracker()

    prev, curr = tracker.update(track_id=1, current_zone="entrance")
    assert prev is None
    assert curr == "entrance"

    prev, curr = tracker.update(track_id=1, current_zone="main_area")
    assert prev == "entrance"
    assert curr == "main_area"


def test_zone_transition_tracker_is_per_track_id():
    tracker = ZoneTransitionTracker()
    tracker.update(track_id=1, current_zone="entrance")
    tracker.update(track_id=2, current_zone="exit")

    prev1, _ = tracker.update(track_id=1, current_zone="main_area")
    prev2, _ = tracker.update(track_id=2, current_zone="main_area")

    assert prev1 == "entrance"
    assert prev2 == "exit"


def test_zone_transition_tracker_bounded_memory():
    tracker = ZoneTransitionTracker(max_tracked_ids=3)
    for track_id in range(1, 6):  # 5 distinct ids, cap is 3
        tracker.update(track_id=track_id, current_zone="zone_a")

    assert len(tracker._last_zone) <= 3
