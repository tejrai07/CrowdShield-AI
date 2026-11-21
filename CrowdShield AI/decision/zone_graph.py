"""Zone graph for the K1 decision engine.

Adjacency is EXPLICITLY CONFIGURED (ZoneConfig.neighbors), never inferred
from polygon geometry. neighbors is a per-zone (directed) list, so a
one-way relationship is supported by omitting the reverse direction.
Candidate ordering is the declared config order (deterministic).
"""
from __future__ import annotations

from config.schema import ZoneConfig


class ZoneGraph:
    def __init__(self, zones: list[ZoneConfig]) -> None:
        self._zones_by_id: dict[str, ZoneConfig] = {z.id: z for z in zones}
        self._validate()

    def _validate(self) -> None:
        known = set(self._zones_by_id)
        for zone in self._zones_by_id.values():
            unknown = set(zone.neighbors) - known
            if unknown:
                raise ValueError(
                    f"ZoneGraph: zone {zone.id!r} references unknown neighbor(s) {sorted(unknown)}"
                )

    @property
    def zone_ids(self) -> list[str]:
        return list(self._zones_by_id)

    def get_zone_config(self, zone_id: str) -> ZoneConfig | None:
        return self._zones_by_id.get(zone_id)

    def neighbors(self, zone_id: str) -> list[str]:
        zone = self._zones_by_id.get(zone_id)
        return list(zone.neighbors) if zone else []

    def is_direct_neighbor(self, source_zone_id: str, candidate_zone_id: str) -> bool:
        return candidate_zone_id in self.neighbors(source_zone_id)

    def routing_cost(self, source_zone_id: str, destination_zone_id: str) -> float:
        """Configured edge cost (default 1.0) for a direct neighbor; inf
        if not a configured direct neighbor. No multi-hop pathfinding."""
        source = self._zones_by_id.get(source_zone_id)
        if source is None or destination_zone_id not in source.neighbors:
            return float("inf")
        return source.neighbor_costs.get(destination_zone_id, 1.0)
