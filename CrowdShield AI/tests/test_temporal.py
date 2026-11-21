"""Unit tests for features/temporal.py.

All synthetic — no YOLO/ByteTrack/video. `ZoneMetrics` instances are
constructed directly with hand-picked values to make expected slopes,
rates, and window contents exactly computable by hand.
"""
from __future__ import annotations

import pytest

from features.crowd_metrics import ZoneMetrics
from features.temporal import TemporalFeatureBuilder, ZoneTemporalTracker, _linear_regression_slope


def _metrics(
    t: float,
    density: float = 0.0,
    speed: float | None = None,
    direction_variance: float | None = None,
    inflow: int = 0,
    outflow: int = 0,
    zone_id: str = "zone_a",
) -> ZoneMetrics:
    return ZoneMetrics(
        zone_id=zone_id,
        timestamp_s=t,
        people_count=0,
        density=density,
        avg_speed_px_s=speed,
        speed_variance=None,
        flow_direction_deg=None,
        direction_variance=direction_variance,
        inflow_count=inflow,
        outflow_count=outflow,
    )


def _tracker(threshold: float = 0.05) -> ZoneTemporalTracker:
    return ZoneTemporalTracker("zone_a", [5, 15, 30, 60], density_growth_threshold=threshold)


# --- window inclusion / boundary --------------------------------------------

def test_boundary_observation_exactly_at_window_edge_is_included():
    tracker = _tracker()
    tracker.add_observation(_metrics(0.0, density=0.5))
    tracker.add_observation(_metrics(5.0, density=0.5))
    features = tracker.compute_features(timestamp_s=5.0)
    assert features.windows[5].sample_count == 2  # t=0 is exactly 5s old -> included


def test_observation_just_past_window_edge_is_excluded():
    tracker = _tracker()
    tracker.add_observation(_metrics(0.0, density=0.5))
    tracker.add_observation(_metrics(5.01, density=0.5))
    features = tracker.compute_features(timestamp_s=5.01)
    assert features.windows[5].sample_count == 1  # t=0 is now >5s old


def test_5s_window_excludes_older_observations():
    tracker = _tracker()
    tracker.add_observation(_metrics(0.0, density=0.1))
    tracker.add_observation(_metrics(10.0, density=0.9))
    features = tracker.compute_features(timestamp_s=10.0)
    assert features.windows[5].sample_count == 1
    assert features.windows[5].density_mean == pytest.approx(0.9)
    assert features.windows[15].sample_count == 2


def test_15_30_60s_windows_roll_correctly():
    tracker = _tracker()
    for t in [0, 10, 20, 40, 55]:
        tracker.add_observation(_metrics(float(t), density=0.1))
    features = tracker.compute_features(timestamp_s=55.0)
    # ages at now=55: t=0->55, t=10->45, t=20->35, t=40->15, t=55->0
    assert features.windows[15].sample_count == 2   # ages 15, 0 (t=40, t=55)
    assert features.windows[30].sample_count == 2    # ages 15, 0 only (t=20's age=35 > 30)
    assert features.windows[60].sample_count == 5     # all ages <= 60


# --- eviction / bounded memory ------------------------------------------------

def test_old_observations_are_evicted():
    tracker = _tracker()
    tracker.add_observation(_metrics(0.0))
    tracker.add_observation(_metrics(100.0))  # far beyond the 60s max window
    assert len(tracker._history) == 1  # t=0 must have been evicted


def test_memory_bounded_regardless_of_duration():
    tracker = _tracker()
    for t in range(0, 1000):
        tracker.add_observation(_metrics(float(t)))
    assert len(tracker._history) <= 61  # ~60s of 1Hz samples, never grows further


# --- density slope --------------------------------------------------------------

def test_density_slope_increasing_sequence_is_positive():
    tracker = _tracker()
    for t in range(0, 6):
        tracker.add_observation(_metrics(float(t), density=t * 0.1))
    features = tracker.compute_features(timestamp_s=5.0)
    assert features.windows[5].density_slope == pytest.approx(0.1, abs=1e-6)


def test_density_slope_decreasing_sequence_is_negative():
    tracker = _tracker()
    for t in range(0, 6):
        tracker.add_observation(_metrics(float(t), density=1.0 - t * 0.1))
    features = tracker.compute_features(timestamp_s=5.0)
    assert features.windows[5].density_slope == pytest.approx(-0.1, abs=1e-6)


def test_density_slope_stable_sequence_is_near_zero():
    tracker = _tracker()
    for t in range(0, 6):
        tracker.add_observation(_metrics(float(t), density=0.5))
    features = tracker.compute_features(timestamp_s=5.0)
    assert features.windows[5].density_slope == pytest.approx(0.0, abs=1e-9)


# --- speed slope -----------------------------------------------------------------

def test_speed_slope_slowing_sequence_is_negative():
    tracker = _tracker()
    for t in range(0, 6):
        tracker.add_observation(_metrics(float(t), speed=20.0 - t * 2.0))
    features = tracker.compute_features(timestamp_s=5.0)
    assert features.windows[5].speed_slope == pytest.approx(-2.0, abs=1e-6)


def test_speed_slope_none_with_no_valid_speed_observations():
    tracker = _tracker()
    for t in range(0, 6):
        tracker.add_observation(_metrics(float(t), speed=None))
    features = tracker.compute_features(timestamp_s=5.0)
    assert features.windows[5].speed_slope is None
    assert features.windows[5].speed_mean is None


# --- direction variance trend -----------------------------------------------------

def test_direction_variance_increasing_dispersion_positive_slope():
    tracker = _tracker()
    for t in range(0, 6):
        tracker.add_observation(_metrics(float(t), direction_variance=t * 0.1))
    features = tracker.compute_features(timestamp_s=5.0)
    assert features.windows[5].direction_variance_slope == pytest.approx(0.1, abs=1e-6)


# --- linear regression helper ------------------------------------------------------

def test_regression_slope_insufficient_points_returns_none():
    assert _linear_regression_slope([]) is None
    assert _linear_regression_slope([(0.0, 1.0)]) is None


def test_regression_slope_identical_x_returns_none():
    assert _linear_regression_slope([(1.0, 5.0), (1.0, 9.0)]) is None


def test_regression_slope_known_line():
    # y = 2x + 1
    points = [(0.0, 1.0), (1.0, 3.0), (2.0, 5.0), (3.0, 7.0)]
    assert _linear_regression_slope(points) == pytest.approx(2.0, abs=1e-9)


# --- inflow / outflow rate -------------------------------------------------------

def test_inflow_rate_known_transitions():
    tracker = _tracker()
    tracker.add_observation(_metrics(0.0, inflow=1))
    tracker.add_observation(_metrics(1.0, inflow=1))
    tracker.add_observation(_metrics(2.0, inflow=1))
    tracker.add_observation(_metrics(4.0, inflow=1))
    features = tracker.compute_features(timestamp_s=4.0)
    # total_inflow=4, span=4.0-0.0=4.0 -> rate=1.0 people/s
    assert features.windows[5].inflow_rate == pytest.approx(1.0)


def test_outflow_rate_known_transitions():
    tracker = _tracker()
    tracker.add_observation(_metrics(0.0, outflow=2))
    tracker.add_observation(_metrics(4.0, outflow=2))
    features = tracker.compute_features(timestamp_s=4.0)
    assert features.windows[5].outflow_rate == pytest.approx(1.0)  # 4 outflow / 4s


def test_net_flow_rate_correct():
    tracker = _tracker()
    tracker.add_observation(_metrics(0.0, inflow=3, outflow=1))
    tracker.add_observation(_metrics(4.0, inflow=1, outflow=1))
    features = tracker.compute_features(timestamp_s=4.0)
    w = features.windows[5]
    assert w.inflow_rate == pytest.approx(1.0)   # 4 inflow / 4s
    assert w.outflow_rate == pytest.approx(0.5)  # 2 outflow / 4s
    assert w.net_flow_rate == pytest.approx(0.5)


def test_rate_none_with_single_observation():
    tracker = _tracker()
    tracker.add_observation(_metrics(0.0, inflow=1))
    features = tracker.compute_features(timestamp_s=0.0)
    assert features.windows[5].inflow_rate is None
    assert features.windows[5].outflow_rate is None
    assert features.windows[5].net_flow_rate is None


# --- bottleneck pressure --------------------------------------------------------

def test_bottleneck_pressure_correct_rolling_calculation():
    tracker = _tracker()
    tracker.add_observation(_metrics(0.0, inflow=5, outflow=1))
    tracker.add_observation(_metrics(2.0, inflow=2, outflow=1))
    features = tracker.compute_features(timestamp_s=2.0)
    # cumulative inflow=7, outflow=2 -> max(0, 7-2)=5
    assert features.windows[5].bottleneck_pressure == pytest.approx(5.0)


def test_bottleneck_pressure_floored_at_zero():
    tracker = _tracker()
    tracker.add_observation(_metrics(0.0, inflow=1, outflow=10))
    features = tracker.compute_features(timestamp_s=0.0)
    assert features.windows[5].bottleneck_pressure == 0.0  # not negative


def test_bottleneck_pressure_available_with_single_sample():
    tracker = _tracker()
    tracker.add_observation(_metrics(0.0, inflow=3, outflow=0))
    features = tracker.compute_features(timestamp_s=0.0)
    assert features.windows[5].bottleneck_pressure == pytest.approx(3.0)


# --- surge flag --------------------------------------------------------------------

def test_surge_flag_true_when_slope_exceeds_threshold():
    tracker = _tracker(threshold=0.05)
    for t in range(0, 6):
        tracker.add_observation(_metrics(float(t), density=t * 0.2))  # slope=0.2 > 0.05
    features = tracker.compute_features(timestamp_s=5.0)
    assert features.surge_flag is True


def test_surge_flag_false_when_slope_below_threshold():
    tracker = _tracker(threshold=0.5)
    for t in range(0, 6):
        tracker.add_observation(_metrics(float(t), density=t * 0.01))  # slope=0.01 < 0.5
    features = tracker.compute_features(timestamp_s=5.0)
    assert features.surge_flag is False


def test_surge_flag_false_during_warmup_with_insufficient_data():
    tracker = _tracker(threshold=0.01)
    tracker.add_observation(_metrics(0.0, density=0.5))  # only 1 sample -> slope None
    features = tracker.compute_features(timestamp_s=0.0)
    assert features.surge_flag is False


# --- warm-up marking ------------------------------------------------------------

def test_incomplete_windows_marked_not_mature():
    tracker = _tracker()
    tracker.add_observation(_metrics(0.0, density=0.5))
    tracker.add_observation(_metrics(3.0, density=0.5))
    features = tracker.compute_features(timestamp_s=3.0)
    assert features.windows[5].is_mature is False
    assert features.windows[5].coverage_s == pytest.approx(3.0)
    assert features.windows[60].is_mature is False


def test_mature_window_when_coverage_reaches_window_size():
    tracker = _tracker()
    tracker.add_observation(_metrics(0.0, density=0.5))
    tracker.add_observation(_metrics(5.0, density=0.5))
    features = tracker.compute_features(timestamp_s=5.0)
    assert features.windows[5].is_mature is True


# --- irregular timestamps --------------------------------------------------------

def test_irregular_timestamps_produce_correct_rates_and_slopes():
    tracker = _tracker()
    # Irregular intervals: 0, 0.7, 1.9, 3.2
    tracker.add_observation(_metrics(0.0, density=0.0, inflow=1))
    tracker.add_observation(_metrics(0.7, density=0.07, inflow=1))
    tracker.add_observation(_metrics(1.9, density=0.19, inflow=1))
    tracker.add_observation(_metrics(3.2, density=0.32, inflow=1))
    features = tracker.compute_features(timestamp_s=3.2)
    w = features.windows[5]
    # density = 0.1 * t exactly -> slope should be ~0.1 despite irregular spacing
    assert w.density_slope == pytest.approx(0.1, abs=1e-6)
    # 4 inflow events over 3.2s actual span (not assumed fixed interval)
    assert w.inflow_rate == pytest.approx(4 / 3.2, abs=1e-6)


# --- empty history ----------------------------------------------------------------

def test_empty_history_no_crash_no_fabricated_values():
    tracker = _tracker()
    features = tracker.compute_features(timestamp_s=0.0)
    for window_seconds in [5, 15, 30, 60]:
        w = features.windows[window_seconds]
        assert w.sample_count == 0
        assert w.coverage_s == 0.0
        assert w.is_mature is False
        assert w.density_mean is None
        assert w.density_slope is None
        assert w.speed_mean is None
        assert w.inflow_rate is None
        assert w.bottleneck_pressure is None
    assert features.surge_flag is False


# --- TemporalFeatureBuilder (multi-zone) ------------------------------------------

def test_builder_routes_metrics_to_correct_zone_tracker():
    builder = TemporalFeatureBuilder(
        zone_ids=["zone_a", "zone_b"], windows_seconds=[5, 15, 30, 60],
        density_growth_threshold=0.05,
    )
    snapshot = {
        "zone_a": _metrics(0.0, density=0.5, zone_id="zone_a"),
        "zone_b": _metrics(0.0, density=0.9, zone_id="zone_b"),
    }
    result = builder.update(snapshot)
    assert result["zone_a"].windows[5].density_mean == pytest.approx(0.5)
    assert result["zone_b"].windows[5].density_mean == pytest.approx(0.9)


def test_builder_ignores_unknown_zone_id():
    builder = TemporalFeatureBuilder(
        zone_ids=["zone_a"], windows_seconds=[5, 15, 30, 60], density_growth_threshold=0.05,
    )
    snapshot = {"unknown_zone": _metrics(0.0, zone_id="unknown_zone")}
    result = builder.update(snapshot)
    assert result == {}
