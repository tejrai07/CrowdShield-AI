"""Data structures for the K1 crowd-diversion decision engine.

K1 is a DECISION-SUPPORT layer, not a certified emergency evacuation
system. It uses ONLY information already produced by D1/E1/F1/H1/I1 — no
new ML model is trained here. CrowdShield does NOT know real-world
evacuation routes, signage, staffing, or human behavior unless those are
explicitly encoded in the venue configuration's zone graph.
"""
from __future__ import annotations

from dataclasses import dataclass

from models.features import ORDINAL_MAP


def label_ordinal(label: str) -> int:
    return ORDINAL_MAP[label]


def label_at_least(label: str, threshold_label: str) -> bool:
    return label_ordinal(label) >= label_ordinal(threshold_label)


@dataclass(frozen=True, slots=True)
class ZoneState:
    """One zone's decision-relevant state at one point in time — the ONLY
    input the K1 engine consumes. Risk fields are optional (None) when
    unavailable (e.g. no I1 future models): K1 degrades gracefully rather
    than fabricating a value."""

    zone_id: str
    enabled: bool
    capacity: float | None
    preferred_utilization_threshold: float | None
    neighbors: list[str]
    neighbor_costs: dict[str, float]
    people_count: int
    density: float
    current_risk_label: str
    future_5m_risk_label: str | None
    future_10m_risk_label: str | None
    surge_flag: bool
    bottleneck_pressure_60s: float | None
    net_flow_rate_60s: float | None

    inputs_reliable: bool = True
    """False when this zone's model inputs are undefined in a way the models were NOT trained on (out-of-distribution): its
    predicted risk is then a diagnostic, not evidence - neither 'safe' nor 'dangerous'."""
    input_note: str | None = None

    @property
    def utilization(self) -> float | None:
        if self.capacity is None or self.capacity <= 0:
            return None
        return self.people_count / self.capacity

    @property
    def worst_known_risk_label(self) -> str:
        candidates = [self.current_risk_label]
        if self.future_5m_risk_label is not None:
            candidates.append(self.future_5m_risk_label)
        if self.future_10m_risk_label is not None:
            candidates.append(self.future_10m_risk_label)
        return max(candidates, key=label_ordinal)


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    risk_component: float
    capacity_component: float
    congestion_component: float
    flow_component: float
    routing_component: float
    total_score: float
    """Lower is better — a penalty score, not a fitness score."""


@dataclass(frozen=True, slots=True)
class RejectedCandidate:
    zone_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class DiversionRecommendation:
    source_zone: str
    destination_zone: str | None
    """None means NO_SAFE_DESTINATION."""
    urgency: str
    """"NORMAL" or "EMERGENCY" (EMERGENCY iff worst known risk is CRITICAL)."""
    confidence: float
    """[0,1] relative margin of the winner over the runner-up — NOT a
    statistical probability. 0.0 for NO_SAFE_DESTINATION."""
    reason: str
    rejected_candidates: list[RejectedCandidate]
    score_breakdown: ScoreBreakdown | None
    timestamp_s: float
    model_horizon_used: str
    trigger_basis: tuple[str, ...] = ()
    """Why this zone is a source: model_current_risk / model_future_risk_5m / model_future_risk_10m (MODEL-DRIVEN) or
    surge_flag / bottleneck_pressure / capacity_rule (rules that do not depend on a risk-model output)."""
    reliability: str = "reliable"
    """'reliable' or 'provisional'. Provisional = every trigger is model-driven AND the source zone's model inputs are
    out-of-distribution: reported for diagnosis, not as a supported conclusion."""
    reliability_note: str | None = None

    @property
    def is_safe_destination_found(self) -> bool:
        return self.destination_zone is not None
