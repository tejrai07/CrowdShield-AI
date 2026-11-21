"""Zone configuration + point-in-polygon assignment.

Converts tracked person positions (from `vision.tracker.Track`) into
configured venue zones. This module owns exactly one responsibility:

    point -> zone_id | None
    track -> zone_id | None

It deliberately does NOT compute density, speed, flow, inflow/outflow, or
risk — those are Phase E/F/H concerns and consume this module's output,
not the other way around.

COORDINATE SYSTEM (D1): image pixel coordinates, origin (0, 0) at the
frame's top-left, x increases rightward, y increases downward — matching
`Detection`/`Track`'s bounding box convention (see vision/detector.py,
vision/tracker.py). Polygons are NOT normalized (0-1) coordinates.
Perspective/real-world calibration is explicitly deferred to a later
phase.

BOUNDARY BEHAVIOR: a point exactly on a zone's polygon boundary counts as
INSIDE that zone. This is checked explicitly (not left to floating-point
luck in the ray-casting test) so the rule holds consistently regardless of
polygon orientation or which edge the point falls on.

OVERLAPPING ZONES: zones are evaluated in configuration order, and the
FIRST enabled zone whose polygon contains the point wins. This also
resolves the common case of two adjacent zones sharing a boundary edge
(e.g. two zones drawn side-by-side) — the earlier zone in the config wins
that shared edge, deterministically.

PERFORMANCE: point-in-polygon here is a plain-Python ray-casting
algorithm — no Shapely/GeoPandas, no per-frame heap allocation beyond a
small list comprehension. This is a hot-path function, called once per
active track per processed frame, so it stays intentionally simple and
allocation-light.
"""
from __future__ import annotations

from dataclasses import dataclass


def _point_on_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float, epsilon: float = 1e-9) -> bool:
    """True if (px, py) lies on the closed segment (x1,y1)-(x2,y2)."""
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if abs(cross) > epsilon:
        return False  # not collinear with this edge

    dot = (px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)
    if dot < -epsilon:
        return False  # before the segment's start

    squared_len = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if dot > squared_len + epsilon:
        return False  # past the segment's end

    return True


def point_in_polygon(px: float, py: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test with an explicit boundary rule.

    Works for arbitrary simple polygons (convex or concave). A point
    exactly on any edge is treated as INSIDE (checked explicitly, before
    the ray-casting pass, so this holds regardless of numerical edge
    cases in the ray test itself).
    """
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if _point_on_segment(px, py, x1, y1, x2, y2):
            return True  # boundary counts as inside

    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > py) != (y2 > py):
            x_intersect = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < x_intersect:
                inside = not inside

    return inside


@dataclass(frozen=True, slots=True)
class Zone:
    """A single configured venue zone, preprocessed once at load time."""

    id: str
    name: str
    polygon: list[tuple[float, float]]
    area_m2: float
    enabled: bool


class ZoneManager:
    """Assigns points/tracks to configured zones.

    Zones are preprocessed once at construction (from `VenueConfig.zones`)
    and never re-parsed per frame. `assign_point`/`assign_track` are the
    only hot-path operations, and both are O(zones) plain-Python checks —
    cheap enough to run once per active track per processed frame.
    """

    def __init__(self, zones: list) -> None:
        # Accepts either config.schema.ZoneConfig objects or already-built
        # Zone objects, so callers can pass `cfg.venue.zones` directly.
        self._zones: list[Zone] = [
            z if isinstance(z, Zone) else Zone(
                id=z.id, name=z.name, polygon=list(z.polygon), area_m2=z.area_m2, enabled=z.enabled
            )
            for z in zones
        ]

    @property
    def zones(self) -> list[Zone]:
        return list(self._zones)

    def get_zone(self, zone_id: str) -> Zone | None:
        for zone in self._zones:
            if zone.id == zone_id:
                return zone
        return None

    def assign_point(self, x: float, y: float) -> str | None:
        """Return the id of the first enabled zone containing (x, y), or
        None if the point falls outside every configured zone."""
        for zone in self._zones:
            if not zone.enabled:
                continue
            if point_in_polygon(x, y, zone.polygon):
                return zone.id
        return None

    def assign_track(self, track) -> str | None:
        """Assign a `vision.tracker.Track` to a zone using its center point."""
        cx, cy = track.center
        return self.assign_point(cx, cy)


class ZoneTransitionTracker:
    """Bounded, session-scoped helper that remembers each track's most
    recent zone, so later phases (Phase E: inflow/outflow) can determine
    previous_zone -> current_zone without re-deriving it themselves.

    D1 does NOT compute inflow/outflow — this class only exposes the raw
    (previous, current) pair per update. Memory is bounded by capping the
    number of remembered track_ids (oldest-updated evicted first), since a
    long-running session could otherwise accumulate one entry per track_id
    ever seen.
    """

    def __init__(self, max_tracked_ids: int = 500) -> None:
        self.max_tracked_ids = max_tracked_ids
        self._last_zone: dict[int, str | None] = {}

    def update(self, track_id: int, current_zone: str | None) -> tuple[str | None, str | None]:
        """Record `current_zone` for `track_id` and return
        (previous_zone, current_zone)."""
        previous_zone = self._last_zone.pop(track_id, None)
        # Re-insert at the end so dict order reflects recency (Python
        # dicts preserve insertion order) — lets us evict the
        # least-recently-updated entry when over capacity.
        self._last_zone[track_id] = current_zone

        if len(self._last_zone) > self.max_tracked_ids:
            oldest_track_id = next(iter(self._last_zone))
            del self._last_zone[oldest_track_id]

        return previous_zone, current_zone

    def get_last_zone(self, track_id: int) -> str | None:
        return self._last_zone.get(track_id)
