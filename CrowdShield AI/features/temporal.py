"""Rolling temporal features + growth/trend detection (Phase F1).

Converts E1's stateless per-snapshot `ZoneMetrics` into short rolling-window
features: `ZoneMetrics(t) -> TemporalFeatureBuilder -> TemporalFeatures`.

F1 is purely temporal — it does not re-run YOLO, ByteTrack, zone
assignment, or geometry. It consumes only E1's public `ZoneMetrics`.

WINDOWS: exactly [5, 15, 30, 60] seconds (config-enforced, see
`config/schema.py::TemporalConfig`) — no arbitrary combinations at this
stage.

BOUNDED MEMORY: each zone keeps one `deque` of timestamped observations,
capped at the largest configured window (60s by default, tied to
`analytics.feature_history_seconds`). Observations older than that are
evicted every time a new one arrives — memory never grows with video
duration.

BOUNDARY RULE: an observation exactly `window_seconds` old is INCLUDED in
that window (`elapsed <= window_seconds`, not `<`). Chosen for consistency
with D1's "boundary counts as inside" convention, and documented/tested
explicitly so the rule is unambiguous.

MISSING DATA, NOT FABRICATED DATA: every statistic that cannot be computed
from available history returns `None` — never a fabricated 0. This
includes: 0 samples in a window (nothing to average), <2 samples for a
slope (can't fit a line through one point), and <2 samples spanning zero
elapsed time for a rate (can't divide by zero time). `sample_count` and
`coverage_s` are exposed on every window so a consumer (H1's future risk
model) can judge whether a window is "mature" (`coverage_s` close to
`window_seconds`) or still warming up.

SLOPES USE LIGHTWEIGHT LEAST-SQUARES REGRESSION over actual timestamps
(not `last - first`, and never an assumed fixed FPS) — see
`_linear_regression_slope()`. This is deliberately simple (no numpy/scipy
dependency) since our window sizes are small (at most a few dozen
samples).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from features.crowd_metrics import ZoneMetrics


@dataclass(frozen=True, slots=True)
class _SnapshotRecord:
    """One retained E1 observation. Only the fields F1 actually needs are
    kept — not a full ZoneMetrics copy — to avoid duplicating large
    objects in bounded history."""

    timestamp_s: float
    density: float
    avg_speed_px_s: float | None
    direction_variance: float | None
    inflow_count: int
    outflow_count: int


@dataclass(frozen=True, slots=True)
class WindowFeatures:
    """Rolling statistics + trend for one temporal window (5/15/30/60s)."""

    window_seconds: int
    sample_count: int
    coverage_s: float
    """Actual time span covered by retained observations within this
    window, in [0, window_seconds]. coverage_s < window_seconds means the
    window is still warming up — see `is_mature`."""
    is_mature: bool
    """True once coverage_s has reached window_seconds (within a small
    epsilon) — i.e. this window has been observing for its full nominal
    duration. False during warm-up. Does NOT mean every statistic below is
    non-None (e.g. speed_mean is still None if no track ever had a valid
    speed measurement in this window)."""

    density_mean: float | None
    density_max: float | None
    density_min: float | None

    speed_mean: float | None
    """Mean of avg_speed_px_s across snapshots with a valid speed in this
    window. UNIT: pixels/second — not calibrated to meters (see E1)."""
    speed_min: float | None

    direction_variance_mean: float | None
    direction_variance_max: float | None

    density_slope: float | None
    """Least-squares slope of density over time in this window.
    UNIT: people/m²/s. Positive = density increasing. None if fewer than
    2 observations, or all observations share the same timestamp."""
    speed_slope: float | None
    """UNIT: pixels/s². Negative = slowing, positive = accelerating. None
    if fewer than 2 valid speed observations in this window."""
    direction_variance_slope: float | None
    """UNIT: 1/s. Positive = movement becoming more dispersed/conflicting.
    None if fewer than 2 valid direction_variance observations."""

    inflow_rate: float | None
    """UNIT: people/second, computed as
    (sum of inflow_count in window) / (elapsed time spanned by the
    window's observations) — NOT divided by the nominal window_seconds,
    since during warm-up that would understate the rate. None if fewer
    than 2 samples or they span zero time."""
    outflow_rate: float | None
    """UNIT: people/second. See inflow_rate for the exact denominator."""
    net_flow_rate: float | None
    """inflow_rate - outflow_rate, people/second. None if either input is
    None."""

    bottleneck_pressure: float | None
    """max(0, cumulative_inflow - cumulative_outflow) within this window —
    a CROWD-FLOW PROXY, not a physically measured pressure. Unlike the
    rate fields, this only needs >=1 sample (it's a bounded cumulative sum,
    not a rate), so it's available slightly earlier during warm-up. None
    only when sample_count is 0."""


@dataclass(frozen=True, slots=True)
class TemporalFeatures:
    """F1 output for one zone at one point in time: one WindowFeatures per
    configured window, plus the derived surge_flag."""

    zone_id: str
    timestamp_s: float
    windows: dict[int, WindowFeatures]
    """Keyed by window_seconds (5, 15, 30, 60)."""
    surge_flag: bool
    """True iff the 5s window's density_slope is not None AND exceeds the
    configured density_growth_threshold. An ENGINEERING TRIGGER for this
    prototype, not a medically/physically validated safety threshold —
    interpretation and tuning belong to a later phase (H1+)."""


def _linear_regression_slope(points: list[tuple[float, float]]) -> float | None:
    """Least-squares slope of y over x. Returns None if fewer than 2
    points, or if all x values are identical (slope undefined — e.g. two
    observations that happen to share a timestamp)."""
    if len(points) < 2:
        return None

    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)

    if denominator == 0:
        return None

    return numerator / denominator


class ZoneTemporalTracker:
    """Bounded rolling history + window feature computation for ONE zone.

    Owns a single deque of `_SnapshotRecord`s, capped at the largest
    configured window. `add_observation()` appends and evicts stale
    entries; `compute_features()` derives all window statistics from
    whatever is currently retained — it never re-touches evicted data.
    """

    def __init__(
        self,
        zone_id: str,
        windows_seconds: list[int],
        density_growth_threshold: float,
        max_history_seconds: float | None = None,
    ) -> None:
        self.zone_id = zone_id
        self.windows_seconds = sorted(windows_seconds)
        self.density_growth_threshold = density_growth_threshold
        # Bounded by the largest window unless explicitly overridden (e.g.
        # to match analytics.feature_history_seconds exactly).
        self.max_history_seconds = max_history_seconds or float(max(self.windows_seconds))
        self._history: deque[_SnapshotRecord] = deque()

    def add_observation(self, metrics: ZoneMetrics) -> None:
        """Record one E1 snapshot and evict anything older than
        max_history_seconds relative to this new observation's timestamp.
        Assumes timestamps arrive in non-decreasing order, consistent with
        a real-time processing loop."""
        record = _SnapshotRecord(
            timestamp_s=metrics.timestamp_s,
            density=metrics.density,
            avg_speed_px_s=metrics.avg_speed_px_s,
            direction_variance=metrics.direction_variance,
            inflow_count=metrics.inflow_count,
            outflow_count=metrics.outflow_count,
        )
        self._history.append(record)

        cutoff = record.timestamp_s - self.max_history_seconds
        while self._history and self._history[0].timestamp_s < cutoff - 1e-9:
            self._history.popleft()

    def compute_features(self, timestamp_s: float | None = None) -> TemporalFeatures:
        """Compute all window features from currently retained history.

        `timestamp_s` defaults to the most recent observation's timestamp
        — pass it explicitly only if you need features "as of" a moment
        that isn't necessarily the latest add_observation() call.
        """
        now = timestamp_s
        if now is None:
            now = self._history[-1].timestamp_s if self._history else 0.0

        windows: dict[int, WindowFeatures] = {}
        for window_seconds in self.windows_seconds:
            windows[window_seconds] = self._compute_window(window_seconds, now)

        return TemporalFeatures(
            zone_id=self.zone_id,
            timestamp_s=now,
            windows=windows,
            surge_flag=self._compute_surge_flag(windows),
        )

    def _compute_window(self, window_seconds: int, now: float) -> WindowFeatures:
        # Boundary rule: an observation exactly window_seconds old is
        # INCLUDED (<=, not <) — see module docstring.
        records = [r for r in self._history if (now - r.timestamp_s) <= window_seconds + 1e-9]
        sample_count = len(records)

        if sample_count == 0:
            return WindowFeatures(
                window_seconds=window_seconds, sample_count=0, coverage_s=0.0, is_mature=False,
                density_mean=None, density_max=None, density_min=None,
                speed_mean=None, speed_min=None,
                direction_variance_mean=None, direction_variance_max=None,
                density_slope=None, speed_slope=None, direction_variance_slope=None,
                inflow_rate=None, outflow_rate=None, net_flow_rate=None,
                bottleneck_pressure=None,
            )

        oldest_ts = records[0].timestamp_s
        coverage_s = now - oldest_ts
        is_mature = coverage_s >= (window_seconds - 1e-9)

        densities = [r.density for r in records]
        density_mean = sum(densities) / len(densities)
        density_max = max(densities)
        density_min = min(densities)

        speeds = [r.avg_speed_px_s for r in records if r.avg_speed_px_s is not None]
        speed_mean = sum(speeds) / len(speeds) if speeds else None
        speed_min = min(speeds) if speeds else None

        dvars = [r.direction_variance for r in records if r.direction_variance is not None]
        direction_variance_mean = sum(dvars) / len(dvars) if dvars else None
        direction_variance_max = max(dvars) if dvars else None

        density_slope = _linear_regression_slope([(r.timestamp_s, r.density) for r in records])
        speed_slope = _linear_regression_slope(
            [(r.timestamp_s, r.avg_speed_px_s) for r in records if r.avg_speed_px_s is not None]
        )
        direction_variance_slope = _linear_regression_slope(
            [(r.timestamp_s, r.direction_variance) for r in records if r.direction_variance is not None]
        )

        total_inflow = sum(r.inflow_count for r in records)
        total_outflow = sum(r.outflow_count for r in records)

        if sample_count >= 2:
            rate_span_s = records[-1].timestamp_s - records[0].timestamp_s
        else:
            rate_span_s = 0.0

        if rate_span_s > 0:
            inflow_rate = total_inflow / rate_span_s
            outflow_rate = total_outflow / rate_span_s
            net_flow_rate = inflow_rate - outflow_rate
        else:
            inflow_rate = None
            outflow_rate = None
            net_flow_rate = None

        bottleneck_pressure = max(0.0, float(total_inflow - total_outflow))

        return WindowFeatures(
            window_seconds=window_seconds,
            sample_count=sample_count,
            coverage_s=coverage_s,
            is_mature=is_mature,
            density_mean=density_mean,
            density_max=density_max,
            density_min=density_min,
            speed_mean=speed_mean,
            speed_min=speed_min,
            direction_variance_mean=direction_variance_mean,
            direction_variance_max=direction_variance_max,
            density_slope=density_slope,
            speed_slope=speed_slope,
            direction_variance_slope=direction_variance_slope,
            inflow_rate=inflow_rate,
            outflow_rate=outflow_rate,
            net_flow_rate=net_flow_rate,
            bottleneck_pressure=bottleneck_pressure,
        )

    def _compute_surge_flag(self, windows: dict[int, WindowFeatures]) -> bool:
        # Uses the fastest-reacting (5s) window deliberately — a longer
        # window would dilute a sudden local spike. If the 5s slope isn't
        # available yet (warm-up, or no valid density history), default
        # to False rather than fabricating a surge signal from no data.
        window_5s = windows.get(5)
        if window_5s is None or window_5s.density_slope is None:
            return False
        return window_5s.density_slope > self.density_growth_threshold


class TemporalFeatureBuilder:
    """Manages one ZoneTemporalTracker per zone and converts a batch of E1
    ZoneMetrics (one per zone, one point in time) into a batch of F1
    TemporalFeatures. This is the component F1's pipeline integration
    calls once per processed frame.
    """

    def __init__(
        self,
        zone_ids: list[str],
        windows_seconds: list[int],
        density_growth_threshold: float,
        max_history_seconds: float | None = None,
    ) -> None:
        self._trackers: dict[str, ZoneTemporalTracker] = {
            zone_id: ZoneTemporalTracker(
                zone_id, windows_seconds, density_growth_threshold, max_history_seconds
            )
            for zone_id in zone_ids
        }

    def update(self, zone_metrics: dict[str, ZoneMetrics]) -> dict[str, TemporalFeatures]:
        """Feed one E1 snapshot per known zone and return this frame's F1
        features for each. Unknown zone_ids (not passed at construction)
        are silently skipped — this mirrors E1's "disabled zones produce
        no entry" behavior rather than raising on a config mismatch."""
        results: dict[str, TemporalFeatures] = {}
        for zone_id, metrics in zone_metrics.items():
            tracker = self._trackers.get(zone_id)
            if tracker is None:
                continue
            tracker.add_observation(metrics)
            results[zone_id] = tracker.compute_features()
        return results
