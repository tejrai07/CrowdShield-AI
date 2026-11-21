"""K1 demo: console demonstration of the crowd-diversion decision engine
against config/venues/k1_demo_venue.yaml. Uses hand-constructed,
illustrative ZoneState scenarios (not a real video) so the decision
LOGIC is what's being demonstrated, reproducibly. See
decision/state_builder.py for wiring K1 into the real live pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.loader import load_config  # noqa: E402
from decision.diversion_engine import DiversionEngine  # noqa: E402
from decision.schemas import ZoneState  # noqa: E402
from decision.zone_graph import ZoneGraph  # noqa: E402
from logging_setup import setup_logging  # noqa: E402


def make_state(graph, zid, risk="SAFE", people=10, f5=None, surge=False):
    zc = graph.get_zone_config(zid)
    return ZoneState(zid, zc.enabled, zc.capacity, zc.preferred_utilization_threshold,
                     zc.neighbors, zc.neighbor_costs, people, people / 40.0, risk,
                     f5, None, surge, None, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="CrowdShield K1 diversion decision engine demo")
    parser.add_argument("--venue", default="config/venues/k1_demo_venue.yaml")
    args = parser.parse_args()

    logger = setup_logging("INFO")
    venue_path = REPO_ROOT / args.venue if not Path(args.venue).is_absolute() else Path(args.venue)
    cfg = load_config(venue_path=venue_path)
    graph = ZoneGraph(cfg.venue.zones)
    engine = DiversionEngine(graph, cfg.diversion)

    states = {
        "entrance": make_state(graph, "entrance", "HIGH", 65),
        "main_hall": make_state(graph, "main_hall", "CRITICAL", 195, surge=True),
        "exit_north": make_state(graph, "exit_north", "MODERATE", 40, f5="HIGH"),
        "exit_south": make_state(graph, "exit_south", "SAFE", 20),
        "overflow_zone": make_state(graph, "overflow_zone", "SAFE", 10),
        "maintenance_closet": make_state(graph, "maintenance_closet", "SAFE", 0),
    }

    for source_id in ["entrance", "main_hall", "exit_north"]:
        source = states[source_id]
        print(f"\nSOURCE: {source_id}")
        print(f"Risk: {source.current_risk_label}" + (f" | +5m: {source.future_5m_risk_label}" if source.future_5m_risk_label else ""))
        print(f"Utilization: {source.utilization:.0%}" if source.utilization is not None else "Utilization: n/a")

        scored, rejected = engine.evaluate_candidates(source, states)
        print("\nCANDIDATES:")
        for cand, bd in scored:
            print(f"  {cand.zone_id:<20} -> score: {bd.total_score:.2f}")
        for rc in rejected:
            print(f"  {rc.zone_id:<20} -> rejected: {rc.reason}")

        rec = engine.recommend_for_source(source, states, timestamp_s=0.0)
        print("\nRECOMMENDATION:")
        if rec.is_safe_destination_found:
            print(f"  Divert toward: {rec.destination_zone}  (urgency: {rec.urgency}, confidence: {rec.confidence:.2f})")
        else:
            print(f"  NO_SAFE_DESTINATION  (urgency: {rec.urgency})")
        print(f"  Reason: {rec.reason}")
        print("-" * 70)

    print(
        "\nNOTE: this is a decision-SUPPORT recommendation based on the venue's "
        "configured zone graph and CrowdShield's risk classifications. "
        "CrowdShield does NOT know real-world evacuation routes, signage, or "
        "staffing beyond what is explicitly configured. This is not a "
        "certified emergency evacuation system."
    )


if __name__ == "__main__":
    main()
