"""Controlled-state PROBE suite for the risk models.

Each probe is a steady operating state (a zone observed for 70 s with identical E1 metrics, so every
temporal window is mature) pushed through the REAL inference path: ZoneMetrics -> TemporalFeatureBuilder
-> inference.pipeline.build_feature_row -> model. E1 semantics are reproduced exactly: no people -> speed and
direction metrics undefined (None); people but zero displacement -> speed 0.0 / variance 0.0 are VALID zeros
while direction is undefined (None).

EXPECTED classes are PROVISIONAL engineering policy, derived from G1's existing density regime
(data/scenarios.yaml: SAFE ~0.5, LOW ~0.8, HIGH ~1.5, CRITICAL >= 2.0 people/m2) - NOT validated safety
thresholds and not real-world ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from features.crowd_metrics import ZoneMetrics
from features.temporal import TemporalFeatureBuilder
from inference.pipeline import build_feature_row

AREA_M2 = 50.0  # same zone area as every G1 scenario
STEPS = 70  # > longest window (60 s) so all windows are mature
SAFE_SET, LOW_SET, DANGEROUS_SET = {"SAFE", "LOW"}, {"SAFE", "LOW", "MODERATE"}, {"HIGH", "CRITICAL"}


@dataclass(frozen=True)
class Probe:
    name: str
    density: float  # people / m2 (people_count = round(density * AREA_M2))
    speed: float | None  # avg speed px/s as E1 would report it (None = undefined)
    dir_var: float | None  # direction variance (None = undefined)
    expected: frozenset
    note: str = ""


PROBES: tuple[Probe, ...] = (
    Probe("empty_zone", 0.0, None, None, frozenset(SAFE_SET), "no people -> all movement metrics undefined"),
    Probe("one_stationary_person", 0.02, 0.0, None, frozenset(SAFE_SET), "valid 0.0 speed, direction undefined"),
    Probe("stationary_low_occupancy", 0.2, 0.0, None, frozenset(SAFE_SET), "10 people standing"),
    Probe("stationary_moderate", 0.8, 0.0, None, frozenset(LOW_SET), "standing queue at G1's LOW density"),
    Probe("dense_stationary_high", 1.6, 0.0, None, frozenset(DANGEROUS_SET), "G1 HIGH-regime density, nobody moving"),
    Probe("dense_stationary_critical", 2.4, 0.0, None, frozenset(DANGEROUS_SET), "G1 CRITICAL-regime density, nobody moving"),
    Probe("normal_moving", 0.5, 60.0, 0.05, frozenset(SAFE_SET), "G1 NORMAL_STABLE"),
    Probe("sparse_moving", 0.1, 50.0, 0.05, frozenset(SAFE_SET), "5 people walking"),
    Probe("moving_dense_high", 1.6, 45.0, 0.1, frozenset(DANGEROUS_SET), "dangerous density at normal speed"),
    Probe("low_speed_dangerous_congestion", 1.6, 8.0, 0.1, frozenset(DANGEROUS_SET), "dense and slow but moving"),
    Probe("low_speed_safe", 0.3, 8.0, 0.3, frozenset(SAFE_SET), "slow strolling, low density"),
    # diagnostic twins: SAME occupancy, but direction defined - isolates the effect of the undefined value
    Probe("diag_one_person_direction_defined", 0.02, 0.0, 0.05, frozenset(SAFE_SET), "diagnostic twin of one_stationary_person"),
    Probe("diag_dense_high_direction_defined", 1.6, 0.0, 0.05, frozenset(DANGEROUS_SET), "diagnostic twin of dense_stationary_high"),
)


def probe_row(probe: Probe, steps: int = STEPS) -> dict[str, float]:
    """Feature row for a probe, built exactly as the live pipeline builds it."""
    people = int(round(probe.density * AREA_M2))
    undefined_speed = probe.speed is None
    builder = TemporalFeatureBuilder(["z"], [5, 15, 30, 60], 0.05, 60.0)
    temporal = None
    for t in range(steps):
        metrics = ZoneMetrics("z", float(t), people, people / AREA_M2, probe.speed, None if undefined_speed else 0.0 if probe.speed == 0.0 else 4.0,
                              None if probe.dir_var is None else 90.0, probe.dir_var, 0, 0)
        temporal = builder.update({"z": metrics})["z"]
    return build_feature_row(metrics, temporal)


def probe_frame(feature_names: list[str], steps: int = STEPS) -> pd.DataFrame:
    rows = {p.name: probe_row(p, steps) for p in PROBES}
    return pd.DataFrame.from_dict(rows, orient="index")[feature_names]


def run_probes(models: dict) -> pd.DataFrame:
    """Predict every probe with {'current'|'future_5m'|'future_10m': model}; returns one row per probe."""
    X = probe_frame(next(iter(models.values())).feature_names)
    out = pd.DataFrame(index=X.index)
    for key, model in models.items():
        out[key] = model.predict_proba(X).idxmax(axis=1)
    out["nan_features"] = X.isna().sum(axis=1)
    out["expected"] = [("/".join(sorted(p.expected))) for p in PROBES]
    for key in models:
        out[f"{key}_ok"] = [out.loc[p.name, key] in p.expected for p in PROBES]
    return out
