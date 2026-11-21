"""Tests for the K1 crowd-diversion decision engine.

Covers scenarios A-L against the real config/venues/k1_demo_venue.yaml
graph, loaded through the real config system (no second config mechanism).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from config.loader import load_config
from config.schema import DiversionConfig, VenueConfig, ZoneConfig
from decision.diversion_engine import DiversionEngine, is_diversion_source
from decision.schemas import ZoneState, label_at_least, label_ordinal
from decision.zone_graph import ZoneGraph

REPO_ROOT = Path(__file__).parent.parent
K1_VENUE_PATH = REPO_ROOT / "config" / "venues" / "k1_demo_venue.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(venue_path=K1_VENUE_PATH)


@pytest.fixture(scope="module")
def graph(cfg):
    return ZoneGraph(cfg.venue.zones)


@pytest.fixture(scope="module")
def engine(graph, cfg):
    return DiversionEngine(graph, cfg.diversion)


def make_state(graph, zid, risk="SAFE", people=10, f5=None, f10=None,
              surge=False, bottleneck=None, net_flow=None):
    zc = graph.get_zone_config(zid)
    return ZoneState(zid, zc.enabled, zc.capacity, zc.preferred_utilization_threshold,
                     zc.neighbors, zc.neighbor_costs, people, people / 40.0, risk,
                     f5, f10, surge, bottleneck, net_flow)


# --- config / graph validation ---------------------------------------------------

def test_k1_venue_loads(cfg):
    assert cfg.venue.name == "k1_demo_venue"
    assert len(cfg.venue.zones) == 6


def test_l_invalid_graph_unknown_neighbor_rejected():
    with pytest.raises(Exception):
        VenueConfig(name="bad", zones=[
            ZoneConfig(id="a", name="A", polygon=[(0, 0), (1, 0), (1, 1)], area_m2=10.0, neighbors=["nope"]),
        ])


def test_l_invalid_graph_self_neighbor_rejected():
    with pytest.raises(Exception):
        VenueConfig(name="bad", zones=[
            ZoneConfig(id="a", name="A", polygon=[(0, 0), (1, 0), (1, 1)], area_m2=10.0, neighbors=["a"]),
        ])


def test_l_neighbor_costs_must_match_neighbors():
    with pytest.raises(Exception):
        VenueConfig(name="bad", zones=[
            ZoneConfig(id="a", name="A", polygon=[(0, 0), (1, 0), (1, 1)], area_m2=10.0,
                      neighbors=["b"], neighbor_costs={"c": 2.0}),
            ZoneConfig(id="b", name="B", polygon=[(0, 0), (1, 0), (1, 1)], area_m2=10.0),
        ])


def test_zone_graph_neighbors_deterministic_order(graph):
    assert graph.neighbors("main_hall") == ["entrance", "exit_north", "exit_south"]


def test_zone_graph_routing_cost(graph):
    assert graph.routing_cost("main_hall", "exit_north") == 1.0
    assert graph.routing_cost("main_hall", "exit_south") == 1.5
    assert graph.routing_cost("entrance", "main_hall") == 1.0


# --- F. disconnected zone unavailable --------------------------------------------

def test_f_disconnected_zone_never_reachable(graph, engine):
    states = {
        "entrance": make_state(graph, "entrance", "CRITICAL", 70),
        "main_hall": make_state(graph, "main_hall", "CRITICAL", 190),
        "exit_north": make_state(graph, "exit_north", "SAFE", 5),  # safe but not entrance's neighbor
    }
    rec = engine.recommend_for_source(states["entrance"], states, 0.0)
    assert rec.destination_zone != "exit_north"
    assert rec.destination_zone is None
    assert graph.routing_cost("entrance", "exit_north") == float("inf")


# --- A. basic connected-zone diversion -----------------------------------------

def test_a_basic_diversion(engine, graph):
    states = {"entrance": make_state(graph, "entrance", "HIGH", 60),
              "main_hall": make_state(graph, "main_hall", "SAFE", 20)}
    rec = engine.recommend_for_source(states["entrance"], states, 0.0)
    assert rec.destination_zone == "main_hall"
    assert rec.is_safe_destination_found
    assert rec.urgency == "NORMAL"


# --- B. nearest rejected due to risk / D. safer farther zone selected -----------

def test_b_d_safer_farther_zone_wins(engine, graph):
    states = {
        "main_hall": make_state(graph, "main_hall", "CRITICAL", 190),
        "entrance": make_state(graph, "entrance", "SAFE", 20),
        "exit_north": make_state(graph, "exit_north", "MODERATE", 10),  # closer, cost 1.0
        "exit_south": make_state(graph, "exit_south", "SAFE", 10),       # farther, cost 1.5, safer
    }
    rec = engine.recommend_for_source(states["main_hall"], states, 0.0)
    assert rec.destination_zone == "exit_south"
    assert rec.urgency == "EMERGENCY"


# --- C. nearest rejected due to capacity -----------------------------------------

def test_c_capacity_rejection(engine, graph):
    states = {
        "main_hall": make_state(graph, "main_hall", "CRITICAL", 190),
        "entrance": make_state(graph, "entrance", "SAFE", 79),  # capacity 80
        "exit_north": make_state(graph, "exit_north", "SAFE", 10),
        "exit_south": make_state(graph, "exit_south", "SAFE", 10),
    }
    rec = engine.recommend_for_source(states["main_hall"], states, 0.0)
    assert rec.destination_zone != "entrance"
    assert "entrance" in {rc.zone_id for rc in rec.rejected_candidates}


def test_future_risk_rejection(engine, graph):
    states = {
        "main_hall": make_state(graph, "main_hall", "CRITICAL", 190),
        "entrance": make_state(graph, "entrance", "SAFE", 20, f5="HIGH"),
        "exit_north": make_state(graph, "exit_north", "SAFE", 10),
        "exit_south": make_state(graph, "exit_south", "SAFE", 10),
    }
    rec = engine.recommend_for_source(states["main_hall"], states, 0.0)
    assert rec.destination_zone != "entrance"
    reason = next(rc.reason for rc in rec.rejected_candidates if rc.zone_id == "entrance")
    assert "+5m" in reason


# --- E. disabled zone rejected -----------------------------------------------------

def test_e_disabled_zone_rejected(graph, cfg):
    zones = list(cfg.venue.zones)
    by_id = {z.id: z for z in zones}
    modified = by_id["exit_north"].model_copy(update={"neighbors": ["main_hall", "maintenance_closet"]})
    zones = [modified if z.id == "exit_north" else z for z in zones]
    local_graph = ZoneGraph(zones)
    local_engine = DiversionEngine(local_graph, cfg.diversion)

    states = {
        "exit_north": make_state(local_graph, "exit_north", "HIGH", 90),
        "main_hall": make_state(local_graph, "main_hall", "CRITICAL", 190),
        "maintenance_closet": make_state(local_graph, "maintenance_closet", "SAFE", 0),
    }
    rec = local_engine.recommend_for_source(states["exit_north"], states, 0.0)
    assert rec.destination_zone != "maintenance_closet"
    reason = next(rc.reason for rc in rec.rejected_candidates if rc.zone_id == "maintenance_closet")
    assert "disabled" in reason.lower()


# --- G. all candidates rejected -> explicit no-safe-destination -----------------

def test_g_no_safe_destination(engine, graph):
    states = {"entrance": make_state(graph, "entrance", "HIGH", 60),
              "main_hall": make_state(graph, "main_hall", "CRITICAL", 190)}
    rec = engine.recommend_for_source(states["entrance"], states, 0.0)
    assert rec.destination_zone is None
    assert not rec.is_safe_destination_found
    assert rec.confidence == 0.0
    assert rec.score_breakdown is None
    assert len(rec.rejected_candidates) >= 1


# --- H. deterministic tie-breaking -----------------------------------------------

def test_h_deterministic_repeated_calls(engine, graph):
    states = {"main_hall": make_state(graph, "main_hall", "HIGH", 100),
              "entrance": make_state(graph, "entrance", "SAFE", 10),
              "exit_north": make_state(graph, "exit_north", "SAFE", 10),
              "exit_south": make_state(graph, "exit_south", "SAFE", 10)}
    r1 = engine.recommend_for_source(states["main_hall"], states, 0.0)
    r2 = engine.recommend_for_source(states["main_hall"], states, 0.0)
    assert r1.destination_zone == r2.destination_zone
    assert r1.score_breakdown.total_score == r2.score_breakdown.total_score


# --- I. score breakdown correctness -----------------------------------------------

def test_i_score_breakdown_matches_weights(engine, graph, cfg):
    states = {"entrance": make_state(graph, "entrance", "HIGH", 60),
              "main_hall": make_state(graph, "main_hall", "SAFE", 20)}
    rec = engine.recommend_for_source(states["entrance"], states, 0.0)
    bd = rec.score_breakdown
    expected = (cfg.diversion.weight_risk * bd.risk_component + cfg.diversion.weight_capacity * bd.capacity_component
                + cfg.diversion.weight_congestion * bd.congestion_component
                + cfg.diversion.weight_flow * bd.flow_component + cfg.diversion.weight_routing * bd.routing_component)
    assert bd.total_score == pytest.approx(expected)


# --- J. multi-source/candidate behavior -------------------------------------------

def test_j_multi_source_no_destination_overload(engine, graph):
    states = {
        "entrance": make_state(graph, "entrance", "HIGH", 70),
        "exit_north": make_state(graph, "exit_north", "HIGH", 70),  # shares only main_hall as neighbor
        "main_hall": make_state(graph, "main_hall", "SAFE", 110),
        "exit_south": make_state(graph, "exit_south", "SAFE", 5),
        "overflow_zone": make_state(graph, "overflow_zone", "SAFE", 5),
    }
    recs = engine.recommend_all(states, 0.0)
    assert recs["entrance"].destination_zone == "main_hall"
    assert recs["exit_north"].destination_zone is None  # main_hall exhausted by entrance's reservation


def test_recommend_all_no_sources_when_all_safe(engine, graph):
    states = {"entrance": make_state(graph, "entrance", "SAFE", 10),
              "main_hall": make_state(graph, "main_hall", "SAFE", 20)}
    assert engine.recommend_all(states, 0.0) == {}


# --- K. missing/NaN metrics -----------------------------------------------------

def test_k_missing_neighbor_state_handled(engine, graph):
    states = {"entrance": make_state(graph, "entrance", "HIGH", 60)}  # main_hall's state absent
    rec = engine.recommend_for_source(states["entrance"], states, 0.0)
    assert rec.destination_zone is None
    assert any("no current state" in rc.reason for rc in rec.rejected_candidates)


def test_k_nan_bottleneck_pressure_does_not_crash(engine, graph):
    states = {
        "entrance": make_state(graph, "entrance", "HIGH", 60),
        "main_hall": make_state(graph, "main_hall", "SAFE", 20, bottleneck=float("nan")),
    }
    rec = engine.recommend_for_source(states["entrance"], states, 0.0)
    assert rec.destination_zone == "main_hall"  # NaN degrades to 0.0, doesn't crash or propagate


def test_k_missing_future_labels_handled(engine, graph):
    states = {"entrance": make_state(graph, "entrance", "HIGH", 60),
              "main_hall": make_state(graph, "main_hall", "SAFE", 20)}
    rec = engine.recommend_for_source(states["entrance"], states, 0.0)
    assert rec.model_horizon_used == "current"


# --- source-zone trigger logic --------------------------------------------------

def test_high_density_alone_does_not_trigger(graph, cfg):
    state = make_state(graph, "main_hall", "SAFE", people=190)
    assert not is_diversion_source(state, cfg.diversion)


def test_current_risk_triggers(graph, cfg):
    assert is_diversion_source(make_state(graph, "main_hall", "HIGH"), cfg.diversion)


def test_future_risk_triggers(graph, cfg):
    assert is_diversion_source(make_state(graph, "main_hall", "SAFE", f5="HIGH"), cfg.diversion)


def test_surge_triggers(graph, cfg):
    assert is_diversion_source(make_state(graph, "main_hall", "SAFE", surge=True), cfg.diversion)


def test_surge_disabled_config(graph):
    config = DiversionConfig(source_surge_triggers=False)
    assert not is_diversion_source(make_state(graph, "main_hall", "SAFE", surge=True), config)


def test_bottleneck_pressure_triggers(graph):
    config = DiversionConfig(source_bottleneck_pressure_threshold=10.0)
    assert is_diversion_source(make_state(graph, "main_hall", "SAFE", bottleneck=15.0), config)


def test_disabled_zone_never_a_source(graph, cfg):
    zc = graph.get_zone_config("main_hall")
    state = ZoneState("main_hall", False, zc.capacity, zc.preferred_utilization_threshold,
                      zc.neighbors, zc.neighbor_costs, 190, 5.0, "CRITICAL", None, None, True, None, None)
    assert not is_diversion_source(state, cfg.diversion)


# --- emergency mode ---------------------------------------------------------------

def test_emergency_never_relaxes_safety(engine, graph):
    states = {
        "main_hall": make_state(graph, "main_hall", "CRITICAL", 190),
        "exit_north": make_state(graph, "exit_north", "CRITICAL", 10),
        "exit_south": make_state(graph, "exit_south", "HIGH", 10),
    }
    rec = engine.recommend_for_source(states["main_hall"], states, 0.0)
    assert rec.destination_zone is None


def test_non_critical_is_normal_urgency(engine, graph):
    states = {"main_hall": make_state(graph, "main_hall", "HIGH", 100),
              "exit_north": make_state(graph, "exit_north", "SAFE", 10)}
    assert engine.recommend_for_source(states["main_hall"], states, 0.0).urgency == "NORMAL"


# --- reason / explanation ----------------------------------------------------------

def test_reason_grounded_in_actual_values(engine, graph):
    states = {"entrance": make_state(graph, "entrance", "HIGH", 60),
              "main_hall": make_state(graph, "main_hall", "SAFE", 20)}
    rec = engine.recommend_for_source(states["entrance"], states, 0.0)
    assert "HIGH" in rec.reason and "main_hall" in rec.reason and "SAFE" in rec.reason


def test_label_helpers():
    assert label_ordinal("SAFE") == 0 and label_ordinal("CRITICAL") == 4
    assert label_at_least("HIGH", "MODERATE") and not label_at_least("LOW", "MODERATE")
    assert label_at_least("MODERATE", "MODERATE")
