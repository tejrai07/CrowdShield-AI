"""Bridge from real E1/F1/H1/I1 outputs to K1's ZoneState. Pure field mapping —
no metric or risk math is duplicated here."""
from __future__ import annotations

from config.schema import ZoneConfig
from decision.schemas import ZoneState
from features.crowd_metrics import ZoneMetrics
from features.temporal import TemporalFeatures


def build_zone_state(zone_config: ZoneConfig, metrics: ZoneMetrics, temporal: TemporalFeatures | None,
                     current_risk_label: str, future_5m_risk_label: str | None = None,
                     future_10m_risk_label: str | None = None, inputs_reliable: bool = True,
                     input_note: str | None = None) -> ZoneState:
    """`temporal` may be None (early warm-up): surge/bottleneck/flow degrade to
    not-available defaults rather than crashing."""
    w60 = temporal.windows.get(60) if temporal is not None else None
    return ZoneState(
        zone_id=zone_config.id, enabled=zone_config.enabled, capacity=zone_config.capacity,
        preferred_utilization_threshold=zone_config.preferred_utilization_threshold,
        neighbors=list(zone_config.neighbors), neighbor_costs=dict(zone_config.neighbor_costs),
        people_count=metrics.people_count, density=metrics.density,
        current_risk_label=current_risk_label, future_5m_risk_label=future_5m_risk_label,
        future_10m_risk_label=future_10m_risk_label,
        surge_flag=temporal.surge_flag if temporal is not None else False,
        bottleneck_pressure_60s=w60.bottleneck_pressure if w60 is not None else None,
        net_flow_rate_60s=w60.net_flow_rate if w60 is not None else None,
        inputs_reliable=inputs_reliable, input_note=input_note,
    )
