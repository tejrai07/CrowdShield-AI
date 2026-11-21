"""Per-zone crowd metrics (Phase E1).

Converts C1 tracks/trajectories + D1 zone assignments into a per-zone
snapshot: `tracks + zone assignments + transition events -> ZoneMetrics`.

This module knows nothing about YOLO, Ultralytics, ByteTrack internals,
ML models, risk thresholds, or dashboards — it consumes only the public
C1 (`vision.tracker.Track`, plain trajectory tuples) and D1
(`features.zones.ZoneManager`, `ZoneTransitionTracker`) abstractions.

E1 IS STATELESS PER SNAPSHOT (aside from the bounded, already-existing
`ZoneTransitionTracker`): every `compute_snapshot()` call produces
independent metrics from the tracks/trajectories passed in. Rolling
windows, trend detection, and accumulated bottleneck pressure are
explicitly deferred to Phase F1 — this keeps the stateless/temporal
boundary clean.

PIXEL SPACE, NOT METERS: C1 trajectories are pixel coordinates, and no
camera calibration exists yet. Speed here is `avg_speed_px_s`
(pixels/second) — NEVER labeled or treated as m/s. Only `density`
(people_count / area_m2) has a real-world unit at this stage, since
`area_m2` is an explicitly configured physical value (D1), unlike pixel
trajectories.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from features.zones import ZoneManager, ZoneTransitionTracker


@dataclass(frozen=True, slots=True)
class ZoneMetrics:
    """A single zone's crowd-state snapshot at one point in time.

    Optional fields are None when there isn't enough data to compute them
    (e.g. a zone with no trajectory history) — never a fabricated 0 or
    other placeholder value, per the "do not fabricate speed" requirement.
    """

    zone_id: str
    timestamp_s: float
    people_count: int
    density: float
    """people_count / area_m2 — the only metric with a real-world unit."""
    avg_speed_px_s: float | None
    """Mean instantaneous pixel speed of valid tracks in this zone.
    UNIT: pixels/second — NOT meters/second. No camera calibration exists
    yet (see module docstring)."""
    speed_variance: float | None
    """Population variance of per-track pixel speeds. 0.0 convention when
    exactly one valid speed observation exists (trivially zero spread);
    None when zero valid observations exist (no data, not "no spread")."""
    flow_direction_deg: float | None
    """Circular mean direction of valid track movement vectors, degrees,
    range [0, 360). See module docstring for the angle convention.
    Meaningless when `direction_variance` is close to 1.0 (near-total
    dispersion) — always check variance before trusting this value."""
    direction_variance: float | None
    """Circular variance (1 - resultant length R), range [0, 1]. 0 = all
    valid tracks moving in exactly the same direction; 1 = directions
    fully dispersed/cancel out. None when zero valid direction
    observations exist."""
    inflow_count: int
    """Number of tracks that transitioned INTO this zone during this one
    snapshot call (from None or another zone) — a single-snapshot count,
    not an accumulated rate. Temporal aggregation is Phase F1's job."""
    outflow_count: int
    """Number of tracks that transitioned OUT OF this zone during this one
    snapshot call (to None or another zone)."""


def track_speed_px_s(trajectory: list[tuple[float, float, float]]) -> float | None:
    """Instantaneous pixel-space speed from a track's last two trajectory
    points: `distance / elapsed_time`, using actual timestamps (never an
    assumed 1/FPS). Returns None if fewer than 2 points are available —
    we do not fabricate a speed for a track with no history.
    """
    if len(trajectory) < 2:
        return None

    t0, x0, y0 = trajectory[-2]
    t1, x1, y1 = trajectory[-1]
    dt = t1 - t0
    if dt <= 0:
        return None  # non-increasing timestamps — no valid speed measurement

    distance = math.hypot(x1 - x0, y1 - y0)
    return distance / dt


def track_direction_deg(trajectory: list[tuple[float, float, float]]) -> float | None:
    """Instantaneous movement direction from a track's last two trajectory
    points, in degrees, range [0, 360).

    ANGLE CONVENTION (image coordinates: x -> right, y -> down):
        angle = atan2(dy, dx) mod 360
        0°   = moving right   (+x)
        90°  = moving down    (+y, since y increases downward on screen)
        180° = moving left    (-x)
        270° = moving up      (-y)
    Because y increases downward (not the standard math/Cartesian
    up-positive convention), increasing angle from 0 -> 90 -> 180 -> 270
    corresponds to a CLOCKWISE sweep as viewed on screen, not
    counter-clockwise as in standard math convention. This is documented
    here deliberately so no caller accidentally assumes Cartesian
    (y-up) semantics.

    Returns None if fewer than 2 points are available, or if the track is
    exactly stationary (zero displacement) — direction is undefined for
    zero movement, so it's excluded from zone-level aggregation rather
    than reported as an arbitrary 0°.
    """
    if len(trajectory) < 2:
        return None

    t0, x0, y0 = trajectory[-2]
    t1, x1, y1 = trajectory[-1]
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return None  # stationary — direction undefined

    angle = math.degrees(math.atan2(dy, dx)) % 360.0
    return 0.0 if angle >= 360.0 else angle  # guard the [0, 360) boundary against float rounding


def circular_mean_and_variance(angles_deg: list[float]) -> tuple[float | None, float | None]:
    """Circular mean direction and circular variance of a set of angles.

    Standard circular statistics (Fisher, "Statistical Analysis of
    Circular Data"):
        C = mean(cos(theta_i)),  S = mean(sin(theta_i))
        R = sqrt(C^2 + S^2)                    # resultant length, [0, 1]
        mean_direction = atan2(S, C) mod 360    # degrees
        circular_variance = 1 - R               # [0, 1]

    This correctly handles wrap-around (e.g. 359° and 1° average to ~0°,
    not the meaningless arithmetic mean of 180°) and correctly signals
    when directions are too dispersed for the mean to be meaningful
    (variance -> 1 as R -> 0), rather than silently returning an
    arithmetic average that could be an artifact of angle wrap-around or
    opposing vectors canceling out.

    Returns (None, None) for an empty input — no data, not a fabricated
    0°/0.0. A single angle returns (that angle, 0.0): trivially "perfectly
    aligned with itself".
    """
    if not angles_deg:
        return None, None

    radians = [math.radians(a) for a in angles_deg]
    cos_mean = sum(math.cos(r) for r in radians) / len(radians)
    sin_mean = sum(math.sin(r) for r in radians) / len(radians)
    resultant_length = math.hypot(cos_mean, sin_mean)

    mean_direction_deg = math.degrees(math.atan2(sin_mean, cos_mean)) % 360.0
    if mean_direction_deg >= 360.0:
        mean_direction_deg = 0.0  # guard the [0, 360) boundary against float rounding
    circular_variance = 1.0 - resultant_length

    return mean_direction_deg, circular_variance


class CrowdMetricsCalculator:
    """Computes a per-zone `ZoneMetrics` snapshot from active tracks +
    their trajectories, once per call. Stateless aside from the bounded
    `ZoneTransitionTracker` it owns (or is given) for inflow/outflow
    event detection across consecutive calls.
    """

    def __init__(
        self,
        zone_manager: ZoneManager,
        transition_tracker: ZoneTransitionTracker | None = None,
    ) -> None:
        self.zone_manager = zone_manager
        self.transition_tracker = transition_tracker or ZoneTransitionTracker()

    def compute_snapshot(
        self,
        tracks: list,
        trajectories: dict[int, list[tuple[float, float, float]]],
        timestamp_s: float,
    ) -> dict[str, ZoneMetrics]:
        """Compute metrics for every enabled zone.

        Args:
            tracks: active `vision.tracker.Track` objects for this frame.
            trajectories: track_id -> that track's bounded trajectory
                (as returned by `PersonTracker.get_trajectory()`). Passed
                in explicitly (rather than this class holding a
                PersonTracker reference) to keep the metrics layer
                decoupled and easy to unit test with synthetic data.
            timestamp_s: current frame timestamp, stamped onto every
                resulting ZoneMetrics.

        Returns:
            dict of zone_id -> ZoneMetrics, one entry per ENABLED zone
            (disabled zones produce no entry — no active crowd metrics
            are generated for them, per the D1 `enabled` flag).
        """
        tracks_by_zone: dict[str, list] = {}
        inflow_counts: dict[str, int] = {}
        outflow_counts: dict[str, int] = {}

        for track in tracks:
            zone_id = self.zone_manager.assign_track(track)
            previous_zone, current_zone = self.transition_tracker.update(track.track_id, zone_id)

            # Only an ACTUAL change counts as a transition — a track that
            # stays in the same zone across many frames must not generate
            # a transition event on every frame it's re-observed in.
            if previous_zone != current_zone:
                if previous_zone is not None:
                    outflow_counts[previous_zone] = outflow_counts.get(previous_zone, 0) + 1
                if current_zone is not None:
                    inflow_counts[current_zone] = inflow_counts.get(current_zone, 0) + 1

            if zone_id is not None:
                tracks_by_zone.setdefault(zone_id, []).append(track)

        results: dict[str, ZoneMetrics] = {}
        for zone in self.zone_manager.zones:
            if not zone.enabled:
                continue

            zone_tracks = tracks_by_zone.get(zone.id, [])
            people_count = len(zone_tracks)
            density = people_count / zone.area_m2

            speeds: list[float] = []
            directions: list[float] = []
            for track in zone_tracks:
                trajectory = trajectories.get(track.track_id, [])
                speed = track_speed_px_s(trajectory)
                if speed is not None:
                    speeds.append(speed)
                direction = track_direction_deg(trajectory)
                if direction is not None:
                    directions.append(direction)

            if not speeds:
                avg_speed_px_s: float | None = None
                speed_variance: float | None = None
            elif len(speeds) == 1:
                avg_speed_px_s = speeds[0]
                speed_variance = 0.0  # documented convention: single observation -> zero spread
            else:
                avg_speed_px_s = sum(speeds) / len(speeds)
                speed_variance = sum((s - avg_speed_px_s) ** 2 for s in speeds) / len(speeds)

            flow_direction_deg, direction_variance = circular_mean_and_variance(directions)

            results[zone.id] = ZoneMetrics(
                zone_id=zone.id,
                timestamp_s=timestamp_s,
                people_count=people_count,
                density=density,
                avg_speed_px_s=avg_speed_px_s,
                speed_variance=speed_variance,
                flow_direction_deg=flow_direction_deg,
                direction_variance=direction_variance,
                inflow_count=inflow_counts.get(zone.id, 0),
                outflow_count=outflow_counts.get(zone.id, 0),
            )

        return results
