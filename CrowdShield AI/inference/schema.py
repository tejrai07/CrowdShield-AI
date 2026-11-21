"""L1 output schema: constants, JSON-safety helper, and a dependency-free
validator for the documented per-frame result (see README "Phase L1")."""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np

from models.features import LABELS_ORDERED

SCHEMA_VERSION = "l1.0"
PIPELINE_STATUSES = ("ok", "warming_up", "degraded")
ZONE_STATUSES = ("ok", "warmup", "unavailable", "disabled", "no_data")
HORIZON_KEYS = ("current", "future_5m", "future_10m")
RECOMMENDATION_STATUSES = ("RECOMMENDED", "NO_SAFE_DESTINATION")
MODEL_DATA_NOTICE = (
    "Risk models were trained ONLY on synthetic data (Phase G1) and have not been validated on real crowd incidents. "
    "Future-risk (+5m/+10m) predictions have low escalation recall (~12.5% at +5m, ~10.7% at +10m): they recognise "
    "persistence of existing danger far better than they anticipate new escalation. Diversion recommendations are "
    "decision support based on configured zone capacity and adjacency only; CrowdShield does not control crowds and "
    "is not a certified evacuation system. null means unavailable, never zero or safe. Zones in unsupported_input_zones have "
    "undefined model inputs the models were never trained on: their predictions are low-reliability diagnostics, and any "
    "recommendation resting only on them is marked provisional."
)
_TOP_KEYS = (
    "schema_version", "timestamp_s", "frame_id", "pipeline_status", "model_data_notice", "capabilities",
    "zones", "explanations", "diversion_recommendations", "warnings", "incomplete_input_zones", "unsupported_input_zones",
)


def to_json_safe(obj: Any) -> Any:
    """Recursively convert to strictly JSON-safe values: NumPy scalars/arrays
    -> Python, dataclasses -> dicts, NaN/+-Infinity -> None (never emitted)."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, int):
        return obj
    if isinstance(obj, np.ndarray):
        return [to_json_safe(x) for x in obj.tolist()]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_json_safe(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return [to_json_safe(x) for x in sorted(obj, key=str)]
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def find_non_finite(obj: Any, path: str = "$") -> list[str]:
    """Paths of any NaN/Infinity floats in a (possibly un-sanitised) structure."""
    found: list[str] = []
    if isinstance(obj, (float, np.floating)) and not math.isfinite(float(obj)):
        found.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found += find_non_finite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found += find_non_finite(v, f"{path}[{i}]")
    return found


def _check_horizon(entry: Any, where: str, errors: list[str]) -> None:
    if not isinstance(entry, dict):
        errors.append(f"{where}: must be an object")
        return
    status = entry.get("status")
    if status not in ("ok", "unavailable"):
        errors.append(f"{where}.status invalid: {status!r}")
    if status == "ok":
        if entry.get("risk_class") not in LABELS_ORDERED:
            errors.append(f"{where}.risk_class invalid: {entry.get('risk_class')!r}")
        score = entry.get("risk_score")
        if not isinstance(score, (int, float)) or not 0.0 <= score <= 100.0:
            errors.append(f"{where}.risk_score must be a number in [0,100]")
    elif status == "unavailable":
        for key in ("risk_class", "risk_score"):
            if entry.get(key) is not None:
                errors.append(f"{where}.{key} must be null when unavailable")
        if not entry.get("reason"):
            errors.append(f"{where}.reason required when unavailable")


def validate_result(result: Any) -> list[str]:
    """Return a list of schema violations (empty list == valid)."""
    errors: list[str] = []
    if not isinstance(result, dict):
        return ["result must be an object"]
    for key in _TOP_KEYS:
        if key not in result:
            errors.append(f"missing top-level key: {key}")
    if errors:
        return errors
    errors += [f"non-finite number at {p}" for p in find_non_finite(result)]
    if result["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if result["pipeline_status"] not in PIPELINE_STATUSES:
        errors.append(f"pipeline_status invalid: {result['pipeline_status']!r}")
    if not isinstance(result["frame_id"], int) or isinstance(result["frame_id"], bool):
        errors.append("frame_id must be an integer")
    if result["timestamp_s"] is not None and not isinstance(result["timestamp_s"], (int, float)):
        errors.append("timestamp_s must be a number or null")
    if not result["model_data_notice"]:
        errors.append("model_data_notice must be non-empty")
    if not (isinstance(result["warnings"], list) and all(isinstance(w, str) for w in result["warnings"])):
        errors.append("warnings must be a list of strings")
    if not isinstance(result["capabilities"], dict):
        errors.append("capabilities must be an object")
    if not (isinstance(result["incomplete_input_zones"], list) and all(isinstance(z, str) for z in result["incomplete_input_zones"])):
        errors.append("incomplete_input_zones must be a list of zone ids")
    if not (isinstance(result["unsupported_input_zones"], list) and all(isinstance(z, str) for z in result["unsupported_input_zones"])):
        errors.append("unsupported_input_zones must be a list of zone ids")
    if not isinstance(result["zones"], list):
        return errors + ["zones must be a list"]
    zone_ids = set()
    for zone in result["zones"]:
        zid = zone.get("zone_id") if isinstance(zone, dict) else None
        where = f"zones[{zid}]"
        if not zid:
            errors.append("zone entry missing zone_id")
            continue
        zone_ids.add(zid)
        if zone.get("status") not in ZONE_STATUSES:
            errors.append(f"{where}.status invalid: {zone.get('status')!r}")
        for key in ("name", "enabled", "people_count", "capacity", "utilization", "density", "metrics", "temporal", "risk", "input_quality"):
            if key not in zone:
                errors.append(f"{where} missing key: {key}")
        risk = zone.get("risk")
        if zone.get("status") in ("ok", "warmup", "unavailable"):
            if not isinstance(risk, dict):
                errors.append(f"{where}.risk must be an object")
            else:
                for hk in HORIZON_KEYS:
                    _check_horizon(risk.get(hk), f"{where}.risk.{hk}", errors)
        elif risk is not None:
            errors.append(f"{where}.risk must be null for status {zone.get('status')!r}")
        if zone.get("status") == "disabled" and zone.get("people_count") is not None:
            errors.append(f"{where}: disabled zones must report people_count null")
    if not isinstance(result["explanations"], dict):
        errors.append("explanations must be an object")
    else:
        for zid, per_h in result["explanations"].items():
            if zid not in zone_ids:
                errors.append(f"explanations references unknown zone {zid!r}")
            for hk, ex in (per_h or {}).items():
                if hk not in HORIZON_KEYS or not isinstance(ex, dict) or ex.get("status") not in ("ok", "unavailable"):
                    errors.append(f"explanations[{zid}][{hk}] malformed")
                elif ex["status"] == "ok" and not all(k in ex for k in ("predicted_class", "top_positive_features", "top_negative_features")):
                    errors.append(f"explanations[{zid}][{hk}] missing fields")
    if not isinstance(result["diversion_recommendations"], list):
        errors.append("diversion_recommendations must be a list")
    else:
        for rec in result["diversion_recommendations"]:
            if rec.get("status") not in RECOMMENDATION_STATUSES:
                errors.append(f"recommendation status invalid: {rec.get('status')!r}")
            elif (rec["status"] == "RECOMMENDED") != (rec.get("destination_zone") is not None):
                errors.append("destination_zone must be set iff status is RECOMMENDED")
            for key in ("source_zone", "urgency", "reason", "rejected_candidates", "source_inputs_complete",
                        "source_inputs_reliable", "trigger_basis", "reliability"):
                if key not in rec:
                    errors.append(f"recommendation missing key: {key}")
            if rec.get("reliability") not in ("reliable", "provisional"):
                errors.append(f"recommendation reliability invalid: {rec.get('reliability')!r}")
            if rec.get("reliability") == "provisional" and not rec.get("reliability_note"):
                errors.append("provisional recommendation must explain itself (reliability_note)")
    return errors
