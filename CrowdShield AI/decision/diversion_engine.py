"""K1 crowd-diversion decision engine.

Answers "if zone A is becoming unsafe, where should CrowdShield recommend
diverting people?" using ONLY information bundled into ZoneState objects
(from D1/E1/F1/H1/I1). No new ML model. Two distinct phases, never blended:
  1. HARD REJECTION: disabled / at-or-over capacity / above preferred
     utilization / current-or-future risk above the configured maximum.
     A rejected candidate is never scored or selected.
  2. SCORING among survivors: deterministic weighted penalty (lower is
     better) of risk, capacity headroom, congestion, destination
     inflow/surge/bottleneck, and routing cost. Weights are explicit
     engineering priorities (risk >> capacity > congestion > flow >=
     routing), documented in config/schema.py::DiversionConfig, NOT
     empirically tuned.
If every neighbor is rejected: NO_SAFE_DESTINATION (destination_zone=None).
Ties are broken alphabetically by zone_id.
"""
from __future__ import annotations

import dataclasses
import math

from config.schema import DiversionConfig
from decision.schemas import (
    DiversionRecommendation,
    RejectedCandidate,
    ScoreBreakdown,
    ZoneState,
    label_at_least,
    label_ordinal,
)
from decision.zone_graph import ZoneGraph

EMERGENCY_LABEL = "CRITICAL"


def _finite(value: float | None, default: float = 0.0) -> float:
    """None or NaN (e.g. F1 warm-up values) -> default; never propagates NaN into a score."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return float(value)


MODEL_DRIVEN_TRIGGERS = frozenset({"model_current_risk", "model_future_risk_5m", "model_future_risk_10m"})


def source_triggers(state: ZoneState, config: DiversionConfig) -> list[str]:
    """Every configured condition that makes this zone a diversion source (empty list = not a source).
    High density alone never triggers; the capacity rule compares people to the zone's CONFIGURED capacity and does not
    depend on any model output, so undefined movement metrics cannot hide an over-capacity crowd."""
    if not state.enabled:
        return []
    out: list[str] = []
    if label_at_least(state.current_risk_label, config.source_current_risk_threshold):
        out.append("model_current_risk")
    for name, future in (("model_future_risk_5m", state.future_5m_risk_label), ("model_future_risk_10m", state.future_10m_risk_label)):
        if future is not None and label_at_least(future, config.source_future_risk_threshold):
            out.append(name)
    if config.source_surge_triggers and state.surge_flag:
        out.append("surge_flag")
    bp = state.bottleneck_pressure_60s
    if (config.source_bottleneck_pressure_threshold is not None and bp is not None and not math.isnan(bp)
            and bp >= config.source_bottleneck_pressure_threshold):
        out.append("bottleneck_pressure")
    util = config.source_capacity_trigger_utilization
    if util is not None and state.capacity and state.people_count >= util * state.capacity:
        out.append("capacity_rule")
    return out


def is_diversion_source(state: ZoneState, config: DiversionConfig) -> bool:
    """True when ANY configured trigger is met (see `source_triggers`)."""
    return bool(source_triggers(state, config))


def _reject_reason(state: ZoneState, config: DiversionConfig, reserved: float) -> str | None:
    if not state.enabled:
        return "zone is disabled"
    if config.destination_requires_reliable_inputs and not state.inputs_reliable:
        return "model inputs are out-of-distribution, so its predicted risk cannot be used as evidence that it is safe" + (f" ({state.input_note})" if state.input_note else "")
    if label_ordinal(state.current_risk_label) > label_ordinal(config.destination_max_current_risk):
        return (f"current risk {state.current_risk_label} exceeds maximum "
                f"allowed destination risk {config.destination_max_current_risk}")
    for name, label in (("+5m", state.future_5m_risk_label), ("+10m", state.future_10m_risk_label)):
        if label is not None and label_ordinal(label) > label_ordinal(config.destination_max_future_risk):
            return (f"predicted {name} risk {label} exceeds maximum "
                    f"allowed destination future risk {config.destination_max_future_risk}")
    if state.capacity is None:
        return "capacity is not configured/available for this zone"
    effective = state.people_count + reserved
    suffix = f", including {reserved:.0f} already reserved by other diversions" if reserved > 0 else ""
    if effective >= state.capacity:
        return f"at/over capacity ({effective:.0f} of {state.capacity:.0f}{suffix})"
    if state.preferred_utilization_threshold is not None:
        util = effective / state.capacity
        if util > state.preferred_utilization_threshold:
            return (f"utilization {util:.0%} exceeds preferred threshold "
                    f"{state.preferred_utilization_threshold:.0%}{suffix and f' (including {reserved:.0f} reserved)'}")
    return None


def _score_candidate(source: ZoneState, cand: ZoneState, graph: ZoneGraph,
                     config: DiversionConfig, emergency: bool, reserved: float) -> ScoreBreakdown:
    risk = float(label_ordinal(cand.worst_known_risk_label))
    capacity_c = (cand.people_count + reserved) / cand.capacity  # capacity guaranteed non-None post-rejection
    congestion = _finite(cand.density)
    flow = max(0.0, _finite(cand.net_flow_rate_60s)) + 0.5 * _finite(cand.bottleneck_pressure_60s)
    if cand.surge_flag:
        flow += 5.0  # fixed documented penalty, not tuned
    routing = graph.routing_cost(source.zone_id, cand.zone_id)

    if emergency:
        w = (config.emergency_weight_risk, config.emergency_weight_capacity, config.emergency_weight_congestion,
             config.emergency_weight_flow, config.emergency_weight_routing)
    else:
        w = (config.weight_risk, config.weight_capacity, config.weight_congestion,
             config.weight_flow, config.weight_routing)
    total = w[0] * risk + w[1] * capacity_c + w[2] * congestion + w[3] * flow + w[4] * routing
    return ScoreBreakdown(risk, capacity_c, congestion, flow, routing, total)


def _horizon(state: ZoneState) -> str:
    parts = ["current"]
    if state.future_5m_risk_label is not None:
        parts.append("5m")
    if state.future_10m_risk_label is not None:
        parts.append("10m")
    return "+".join(parts)


def _reason(source: ZoneState, winner: ZoneState | None, bd: ScoreBreakdown | None,
            rejected: list[RejectedCandidate], emergency: bool) -> str:
    text = [f"{source.zone_id} has current risk {source.current_risk_label}"
            + (f", +5m risk {source.future_5m_risk_label}" if source.future_5m_risk_label else "")
            + (f", +10m risk {source.future_10m_risk_label}" if source.future_10m_risk_label else "")
            + (" - EMERGENCY mode active." if emergency else ".")]
    text += [f"{rc.zone_id} rejected: {rc.reason}." for rc in rejected]
    if winner is not None and bd is not None:
        util = f"{winner.utilization:.0%} utilization" if winner.utilization is not None else "utilization n/a"
        text.append(f"{winner.zone_id} selected: risk {winner.worst_known_risk_label}, {util}, "
                    f"density {winner.density:.2f} people/m2, score {bd.total_score:.2f} (lower is better).")
    else:
        text.append("No reachable neighbor passed safety filtering - NO_SAFE_DESTINATION.")
    return " ".join(text)


class DiversionEngine:
    """Stateless-per-call engine: all state arrives via ZoneState/reserved-capacity args."""

    def __init__(self, graph: ZoneGraph, config: DiversionConfig) -> None:
        self.graph = graph
        self.config = config

    def evaluate_candidates(self, source: ZoneState, all_states: dict[str, ZoneState],
                            reserved: dict[str, float] | None = None):
        """Returns (scored, rejected): scored = [(ZoneState, ScoreBreakdown)] sorted best-first
        (score, then zone_id); rejected = [RejectedCandidate]. Used by recommend_for_source
        and by the demo to show ALL candidates transparently."""
        reserved = reserved or {}
        emergency = source.worst_known_risk_label == EMERGENCY_LABEL
        scored, rejected = [], []
        for cid in self.graph.neighbors(source.zone_id):
            cand = all_states.get(cid)
            if cand is None:
                rejected.append(RejectedCandidate(cid, "no current state available for this zone"))
                continue
            r = reserved.get(cid, 0.0)
            reason = _reject_reason(cand, self.config, r)
            if reason is not None:
                rejected.append(RejectedCandidate(cid, reason))
                continue
            scored.append((cand, _score_candidate(source, cand, self.graph, self.config, emergency, r)))
        scored.sort(key=lambda p: (p[1].total_score, p[0].zone_id))
        return scored, rejected

    def recommend_for_source(self, source: ZoneState, all_states: dict[str, ZoneState], timestamp_s: float,
                             reserved_capacity: dict[str, float] | None = None) -> DiversionRecommendation:
        """Destination selection (unchanged) plus provenance: WHY the zone is a source and whether that conclusion is supported.
        Policy (deterministic, PROVISIONAL engineering policy): a recommendation is 'provisional' iff the source's model inputs are
        out-of-distribution AND every trigger is model-driven. If any model-independent trigger (capacity_rule, surge_flag,
        bottleneck_pressure) fired, the recommendation rests on that rule and stays 'reliable'. Nothing is suppressed."""
        rec = self._recommend_core(source, all_states, timestamp_s, reserved_capacity)
        basis = tuple(source_triggers(source, self.config))
        independent = [b for b in basis if b not in MODEL_DRIVEN_TRIGGERS]
        if basis and not source.inputs_reliable and not independent:
            note = ("all triggers are model-driven but this zone's model inputs are out-of-distribution"
                    + (f" ({source.input_note})" if source.input_note else "") + "; treat as unverified")
            return dataclasses.replace(rec, trigger_basis=basis, reliability="provisional", reliability_note=note)
        note = (f"model inputs are out-of-distribution; recommendation rests on model-independent rule(s): {', '.join(independent)}"
                if basis and not source.inputs_reliable else None)
        return dataclasses.replace(rec, trigger_basis=basis, reliability="reliable", reliability_note=note)

    def _recommend_core(self, source: ZoneState, all_states: dict[str, ZoneState], timestamp_s: float,
                        reserved_capacity: dict[str, float] | None = None) -> DiversionRecommendation:
        emergency = source.worst_known_risk_label == EMERGENCY_LABEL
        urgency = "EMERGENCY" if emergency else "NORMAL"
        scored, rejected = self.evaluate_candidates(source, all_states, reserved_capacity)
        rejected = rejected[: self.config.max_rejected_candidates_reported]

        if not scored:
            return DiversionRecommendation(source.zone_id, None, urgency, 0.0,
                                           _reason(source, None, None, rejected, emergency),
                                           rejected, None, timestamp_s, _horizon(source))
        winner, bd = scored[0]
        if len(scored) == 1:
            confidence = 1.0
        else:
            runner = scored[1][1].total_score
            confidence = max(0.0, min(1.0, (runner - bd.total_score) / runner)) if runner > 0 else 1.0
        return DiversionRecommendation(source.zone_id, winner.zone_id, urgency, confidence,
                                       _reason(source, winner, bd, rejected, emergency),
                                       rejected, bd, timestamp_s, _horizon(source))

    def recommend_all(self, all_states: dict[str, ZoneState], timestamp_s: float
                      ) -> dict[str, DiversionRecommendation]:
        """Greedy multi-source allocation: sources processed most-severe first (then
        alphabetical); each accepted diversion reserves its source's people_count at the
        destination so later sources can't collectively overload it."""
        sources = [s for s in all_states.values() if is_diversion_source(s, self.config)]
        sources.sort(key=lambda s: (-label_ordinal(s.worst_known_risk_label), s.zone_id))
        reserved: dict[str, float] = {}
        out: dict[str, DiversionRecommendation] = {}
        for src in sources:
            rec = self.recommend_for_source(src, all_states, timestamp_s, reserved)
            out[src.zone_id] = rec
            if rec.destination_zone is not None:
                reserved[rec.destination_zone] = reserved.get(rec.destination_zone, 0.0) + src.people_count
        return out
