"""L1 end-to-end inference pipeline.

frame -> detection -> tracking -> zone assignment + per-zone metrics (E1) ->
temporal features (F1) -> H1/I1 risk -> J1 explanations -> K1 diversion
recommendations -> one JSON-safe result dict per processed frame.

This module is ORCHESTRATION ONLY. It reuses, and does not re-implement:
  vision.detector / vision.tracker, features.crowd_metrics (which performs
  zone assignment and inflow/outflow transitions internally), features.temporal,
  models.risk_model / models.future_risk / models.explainability, and
  decision.state_builder / decision.diversion_engine.
The only new computation is mapping E1/F1 objects into the model's flat
feature row (`build_feature_row`), because no existing helper does it.

HONESTY RULES (enforced and tested):
  * null always means unavailable - missing data is never converted into a
    zero or a SAFE label. A zone whose current prediction is unavailable is
    EXCLUDED from K1 rather than treated as safe.
  * warm-up (temporal windows not yet full), degraded operation, and genuine
    low-risk predictions are reported as distinct states.
  * Models are synthetic-data-only; every result carries `model_data_notice`.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from config.schema import AppConfig
from decision.diversion_engine import DiversionEngine
from decision.state_builder import build_zone_state
from decision.zone_graph import ZoneGraph
from features.crowd_metrics import CrowdMetricsCalculator, ZoneMetrics
from features.temporal import TemporalFeatureBuilder, TemporalFeatures, WindowFeatures
from features.zones import ZoneManager
from inference.schema import HORIZON_KEYS, MODEL_DATA_NOTICE, SCHEMA_VERSION, to_json_safe
from models.explainability import ModelExplainer
from models.features import LABELS_ORDERED, prepare_features
from models.future_risk import HORIZON_SECONDS, MultiHorizonRiskModel, dangerous_probability
from models.risk_model import RiskModel, risk_score_from_proba

logger = logging.getLogger("crowdshield")

_HORIZON_NAME = {"current": "current", "future_5m": "5m", "future_10m": "10m"}
_WINDOW_FIELDS = [f.name for f in dataclasses.fields(WindowFeatures) if f.name != "window_seconds"]
_DANGEROUS = ("HIGH", "CRITICAL")


class PipelineError(RuntimeError):
    """The pipeline cannot start, or has failed unrecoverably."""


def _num(value: Any) -> float:
    """None / NaN / +-Infinity -> NaN (explicitly 'undefined' to the model); bool -> 0/1; every finite number unchanged (a real 0.0 stays 0.0)."""
    if value is None:
        return float("nan")
    if isinstance(value, (bool, np.bool_)):
        return float(int(value))
    number = float(value)
    return number if math.isfinite(number) else float("nan")  # +-Infinity is not a measurement: treat as undefined, never pass it to a model


def build_feature_row(metrics: ZoneMetrics, temporal: TemporalFeatures | None) -> dict[str, float]:
    """Flatten one zone's E1 metrics + F1 temporal features into the flat
    feature row the H1/I1 models were trained on (G1's column naming:
    base fields, then w{N}_{WindowFeatures field}). Missing values are NaN,
    which the HistGradientBoosting models treat natively as 'unavailable'."""
    row = {
        "people_count": _num(metrics.people_count), "density": _num(metrics.density),
        "avg_speed_px_s": _num(metrics.avg_speed_px_s), "speed_variance": _num(metrics.speed_variance),
        "flow_direction_deg": _num(metrics.flow_direction_deg), "direction_variance": _num(metrics.direction_variance),
        "inflow_count": _num(metrics.inflow_count), "outflow_count": _num(metrics.outflow_count),
        "surge_flag": _num(temporal.surge_flag) if temporal is not None else float("nan"),
    }
    windows = temporal.windows if temporal is not None else {}
    for seconds, window in windows.items():
        for name in _WINDOW_FIELDS:
            row[f"w{seconds}_{name}"] = _num(getattr(window, name))
    return row


def _valid_frame(frame: Any) -> bool:
    return isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.size > 0


def _unavailable(seconds: float, reason: str, model_version: str | None = None) -> dict:
    return {"status": "unavailable", "risk_class": None, "risk_score": None, "dangerous_probability": None,
            "class_probabilities": None, "horizon_seconds": seconds, "model_version": model_version, "reason": reason}


def _proba_problem(proba: Any, n_rows: int) -> str | None:
    """Table-level validation of a model's probability output."""
    if not isinstance(proba, pd.DataFrame) or set(proba.columns) != set(LABELS_ORDERED) or len(proba) != n_rows:
        return "invalid model output: probability table has unexpected shape or columns"
    return None


def _row_problem(values: np.ndarray) -> str | None:
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0 + 1e-9):
        return "invalid model output: probabilities missing, non-finite or outside [0, 1]"
    if abs(float(values.sum()) - 1.0) > 1e-3:
        return "invalid model output: probabilities do not sum to 1"
    return None


class InferencePipeline:
    """Holds all long-lived components (loaded once) and turns frames into results."""

    def __init__(self, cfg: AppConfig, detector: Any, tracker: Any, risk_models: dict[str, Any],
                 explainers: dict[str, ModelExplainer] | None = None, init_warnings: list[str] | None = None) -> None:
        if "current" not in risk_models:
            raise PipelineError("a 'current' risk model is required")
        if not any(z.enabled for z in cfg.venue.zones):
            raise PipelineError("venue has no enabled zones - configure zones (e.g. --venue config/venues/example_venue.yaml)")
        self.cfg = cfg
        self.detector, self.tracker = detector, tracker
        self.risk_models = dict(risk_models)
        self.explainers = dict(explainers or {})
        self._zone_configs = list(cfg.venue.zones)
        self.zone_manager = ZoneManager(cfg.venue.zones)
        self.metrics_calculator = CrowdMetricsCalculator(self.zone_manager)
        self.temporal_builder = TemporalFeatureBuilder(
            zone_ids=[z.id for z in self._zone_configs if z.enabled],
            windows_seconds=list(cfg.temporal.windows_seconds),
            density_growth_threshold=cfg.temporal.density_growth_threshold,
            max_history_seconds=cfg.analytics.feature_history_seconds,
        )
        self.graph = ZoneGraph(cfg.venue.zones)
        self.engine = DiversionEngine(self.graph, cfg.diversion)
        self._longest_window = max(cfg.temporal.windows_seconds)
        self._last_timestamp: float | None = None
        self._consecutive_failures = 0
        self._last_classes: dict[str, dict[str, str | None]] = {}
        self._init_warnings = list(init_warnings or [])

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_config(cls, cfg: AppConfig, repo_root: Path | str | None = None, detector: Any = None,
                    tracker: Any = None, risk_models: dict[str, Any] | None = None,
                    explainers: dict[str, ModelExplainer] | None = None) -> "InferencePipeline":
        """Load real components from config; any component can be injected
        (tests do). Fails clearly - never fabricates output - if a required
        component cannot load. Explanations are optional and degrade to a warning."""
        root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
        warnings: list[str] = []
        # Cheap, local checks first: fail fast on missing models BEFORE constructing the detector (which may download weights).
        if risk_models is None:
            risk_models, w = _load_risk_models(_resolve(root, cfg.inference.model_dir))
            warnings += w
        if explainers is None:
            explainers, w = _build_explainers(risk_models, cfg, root)
            warnings += w
        if detector is None:
            try:
                from vision.detector import PersonDetector
                detector = PersonDetector(model=cfg.detection.model, confidence=cfg.detection.confidence,
                                          device=cfg.detection.device, image_size=cfg.detection.image_size)
            except Exception as exc:
                raise PipelineError(f"failed to initialise person detector ({cfg.detection.model}): "
                                    f"{type(exc).__name__}: {exc}") from exc
        if tracker is None:
            try:
                from vision.tracker import PersonTracker
                tracker = PersonTracker(max_age_frames=cfg.tracking.max_age,
                                        trajectory_history_seconds=cfg.analytics.trajectory_history_seconds)
            except Exception as exc:
                raise PipelineError(f"failed to initialise tracker: {type(exc).__name__}: {exc}") from exc
        return cls(cfg, detector, tracker, risk_models, explainers, warnings)

    # ---------------------------------------------------------------- main loop
    def run(self, frame_source: Iterable[Any], max_frames: int | None = None) -> Iterator[dict]:
        """Yield one result per processed frame. `frame_source` yields objects
        with .frame, .source_frame_index, .source_timestamp_s (vision.benchmark.SampledFrame).
        Results are yielded, never accumulated; the source is always closed."""
        if max_frames is not None and max_frames < 0:
            raise ValueError("max_frames must be >= 0")
        iterator = iter(frame_source)
        count = 0
        try:
            for sampled in iterator:
                if max_frames is not None and count >= max_frames:
                    break
                yield self.process_frame(sampled.frame, sampled.source_frame_index, sampled.source_timestamp_s)
                count += 1
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    def process_frame(self, frame: Any, frame_id: int, timestamp_s: float) -> dict:
        warnings = list(self._init_warnings)
        self._init_warnings = []  # one-time init warnings are reported on the first result only

        ts = float(timestamp_s) if isinstance(timestamp_s, (int, float, np.floating, np.integer)) and math.isfinite(float(timestamp_s)) else None
        if ts is None:
            return self._no_data_result(frame_id, None, warnings + ["invalid timestamp - frame skipped"])
        if self._last_timestamp is not None and ts < self._last_timestamp:
            return self._no_data_result(frame_id, ts, warnings + [
                f"timestamp {ts:.3f}s precedes previous {self._last_timestamp:.3f}s - frame skipped (temporal features require non-decreasing time)"])
        if not _valid_frame(frame):
            return self._no_data_result(frame_id, ts, warnings + ["missing or invalid frame (expected non-empty HxWxC array) - frame skipped"])

        try:
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections, ts)
        except Exception as exc:
            self._consecutive_failures += 1
            message = f"detection/tracking failed: {type(exc).__name__}: {exc}"
            if self._consecutive_failures >= self.cfg.inference.max_consecutive_failures:
                raise PipelineError(f"{message} ({self._consecutive_failures} consecutive failures)") from exc
            return self._no_data_result(frame_id, ts, warnings + [message])
        self._consecutive_failures = 0
        self._last_timestamp = ts

        trajectories = {t.track_id: self.tracker.get_trajectory(t.track_id) for t in tracks}
        zone_metrics = self.metrics_calculator.compute_snapshot(tracks, trajectories, ts)
        temporal = self.temporal_builder.update(zone_metrics)

        zone_ids = [z.id for z in self._zone_configs if z.enabled and z.id in zone_metrics]
        X = pd.DataFrame.from_dict({zid: build_feature_row(zone_metrics[zid], temporal.get(zid)) for zid in zone_ids}, orient="index")
        predictions, pred_warnings = self._predict(X)
        warnings += pred_warnings
        quality = self._input_quality(X, zone_metrics)
        incomplete = [zid for zid in zone_ids if not quality[zid]["complete"]]
        unsupported = [zid for zid in zone_ids if not quality[zid]["reliable"]]
        if unsupported:
            warnings.append(
                "model inputs are out-of-distribution for zone(s) " + ", ".join(unsupported) + ": the set of undefined inputs was not seen "
                "in training (or the model carries no training-pattern metadata), so their predictions are LOW-RELIABILITY diagnostics - "
                "reported as computed, never imputed. Recommendations resting only on model output for these zones are marked provisional.")

        statuses: dict[str, str] = {}
        for zid in zone_ids:
            if predictions[zid]["current"]["status"] != "ok":
                statuses[zid] = "unavailable"
            else:
                window = temporal[zid].windows.get(self._longest_window)
                statuses[zid] = "ok" if (window is not None and window.is_mature) else "warmup"

        zone_states = {}
        for zid in zone_ids:
            cur = predictions[zid]["current"]
            if cur["status"] != "ok":
                warnings.append(f"zone {zid} excluded from diversion decisions: current risk unavailable ({cur['reason']})")
                continue
            f5, f10 = predictions[zid]["future_5m"], predictions[zid]["future_10m"]
            zone_states[zid] = build_zone_state(
                next(z for z in self._zone_configs if z.id == zid), zone_metrics[zid], temporal.get(zid), cur["risk_class"],
                f5["risk_class"] if f5["status"] == "ok" else None, f10["risk_class"] if f10["status"] == "ok" else None,
                inputs_reliable=quality[zid]["reliable"], input_note=self._quality_note(quality[zid]))
        for z in self._zone_configs:
            if not z.enabled:
                warnings.append(f"zone {z.id} is disabled and excluded from metrics, prediction and diversion decisions")

        recommendations: list[dict] = []
        recs: dict = {}
        try:
            recs = self.engine.recommend_all(zone_states, ts)
            recommendations = [self._recommendation_entry(r, statuses.get(r.source_zone) == "warmup", quality[r.source_zone]["complete"],
                                                          quality[r.source_zone]["reliable"]) for r in recs.values()]
            warnings += [f"diversion recommendation for zone {r.source_zone} is PROVISIONAL: {r.reliability_note}"
                         for r in recs.values() if r.reliability == "provisional"]
        except Exception as exc:
            warnings.append(f"diversion engine failed: {type(exc).__name__}: {exc}")

        explanations, expl_warnings = self._explain(X, predictions, statuses, set(recs))
        warnings += expl_warnings

        zones_out = [self._zone_entry(z, zone_metrics.get(z.id), temporal.get(z.id), predictions.get(z.id), statuses.get(z.id), quality.get(z.id)) for z in self._zone_configs]
        degraded = any(e["status"] == "unavailable" for zid in zone_ids for e in predictions[zid].values())
        status = "degraded" if degraded else ("warming_up" if "warmup" in statuses.values() else "ok")
        return self._assemble(frame_id, ts, status, zones_out, explanations, recommendations, warnings, incomplete, unsupported)

    # ---------------------------------------------------------------- internals
    def _predict(self, X: pd.DataFrame) -> tuple[dict[str, dict[str, dict]], list[str]]:
        out: dict[str, dict[str, dict]] = {zid: {} for zid in X.index}
        warnings: list[str] = []
        for key in HORIZON_KEYS:
            seconds = HORIZON_SECONDS[_HORIZON_NAME[key]]
            model = self.risk_models.get(key)
            if model is None:
                for zid in X.index:
                    out[zid][key] = _unavailable(seconds, "model not loaded")
                continue
            version = getattr(model, "model_version", None)
            try:
                proba = model.predict_proba(X)
            except Exception as exc:
                reason = f"model error: {type(exc).__name__}: {exc}"
                warnings.append(f"{key} prediction failed for all zones: {reason}")
                for zid in X.index:
                    out[zid][key] = _unavailable(seconds, reason, version)
                continue
            table_problem = _proba_problem(proba, len(X))
            if table_problem:
                warnings.append(f"{key}: {table_problem}")
                for zid in X.index:
                    out[zid][key] = _unavailable(seconds, table_problem, version)
                continue
            values = proba[LABELS_ORDERED].to_numpy(dtype=float)
            scores = risk_score_from_proba(proba)
            danger = dangerous_probability(proba)
            for i, zid in enumerate(X.index):
                problem = _row_problem(values[i])
                if problem is None and not (math.isfinite(float(scores[i])) and 0.0 <= float(scores[i]) <= 100.0):
                    problem = "invalid model output: risk score outside [0, 100]"
                if problem:
                    warnings.append(f"{key} prediction for zone {zid} unavailable: {problem}")
                    out[zid][key] = _unavailable(seconds, problem, version)
                    continue
                out[zid][key] = {
                    "status": "ok", "risk_class": LABELS_ORDERED[int(np.argmax(values[i]))], "risk_score": float(scores[i]),
                    "dangerous_probability": float(danger.iloc[i]),
                    "class_probabilities": {lab: float(values[i][j]) for j, lab in enumerate(LABELS_ORDERED)},
                    "horizon_seconds": seconds, "model_version": version, "reason": None,
                }
        return out, warnings

    def _explain(self, X, predictions, statuses, source_zones):
        mode = self.cfg.inference.explain_mode
        explanations: dict[str, dict] = {}
        warnings: list[str] = []
        for zid, per_h in predictions.items():
            now = {k: (e["risk_class"] if e["status"] == "ok" else None) for k, e in per_h.items()}
            prev = self._last_classes.get(zid)
            self._last_classes[zid] = now
            if mode == "off" or not self.explainers or statuses.get(zid) == "unavailable":
                continue
            wanted = (mode == "all" or zid in source_zones or (prev is not None and prev != now)
                      or any(c in _DANGEROUS for c in now.values() if c))
            if not wanted:
                continue
            for key in HORIZON_KEYS:
                explainer = self.explainers.get(key)
                if explainer is None or per_h[key]["status"] != "ok":
                    continue
                try:
                    ex = explainer.explain(X.loc[[zid]], top_k=self.cfg.inference.explain_top_k)
                except Exception as exc:
                    warnings.append(f"explanation unavailable for zone {zid} ({key}): {type(exc).__name__}: {exc}")
                    explanations.setdefault(zid, {})[key] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
                    continue

                def feat(fc, direction):
                    return {"feature": fc.feature, "description": fc.human_name, "value": fc.value,
                            "reference_value": fc.reference_value, "contribution": fc.contribution, "direction": direction}

                explanations.setdefault(zid, {})[key] = {
                    "status": "ok", "predicted_class": ex.predicted_class, "predicted_probability": ex.predicted_probability,
                    "risk_score": ex.risk_score, "horizon_seconds": ex.horizon_seconds, "model_version": ex.model_version,
                    "top_positive_features": [feat(f, "supports_prediction") for f in ex.top_positive_features],
                    "top_negative_features": [feat(f, "opposes_prediction") for f in ex.top_negative_features],
                }
        return explanations, warnings

    @staticmethod
    def _recommendation_entry(rec, warmup: bool, inputs_complete: bool, inputs_reliable: bool = True) -> dict:
        return {
            "source_zone": rec.source_zone, "destination_zone": rec.destination_zone,
            "status": "RECOMMENDED" if rec.destination_zone is not None else "NO_SAFE_DESTINATION",
            "urgency": rec.urgency, "confidence": rec.confidence, "reason": rec.reason,
            "rejected_candidates": [{"zone_id": c.zone_id, "reason": c.reason} for c in rec.rejected_candidates],
            "score_breakdown": rec.score_breakdown, "model_horizon_used": rec.model_horizon_used,
            "during_warmup": bool(warmup), "source_inputs_complete": bool(inputs_complete),
            "source_inputs_reliable": bool(inputs_reliable), "trigger_basis": list(rec.trigger_basis),
            "reliability": rec.reliability, "reliability_note": rec.reliability_note,
        }

    def _zone_entry(self, zc, metrics, temporal, predictions, status, quality=None) -> dict:
        base = {"zone_id": zc.id, "name": zc.name, "enabled": zc.enabled, "capacity": zc.capacity,
                "people_count": None, "density": None, "utilization": None,
                "metrics": None, "temporal": None, "risk": None, "input_quality": None}
        if not zc.enabled:
            return {**base, "status": "disabled"}
        if metrics is None or predictions is None:
            return {**base, "status": "no_data"}
        windows = temporal.windows if temporal is not None else {}
        return {
            **base, "status": status, "people_count": metrics.people_count, "density": metrics.density,
            "utilization": (metrics.people_count / zc.capacity) if zc.capacity else None,
            "metrics": {"avg_speed_px_s": metrics.avg_speed_px_s, "speed_variance": metrics.speed_variance,
                        "flow_direction_deg": metrics.flow_direction_deg, "direction_variance": metrics.direction_variance,
                        "inflow_count": metrics.inflow_count, "outflow_count": metrics.outflow_count},
            "temporal": {
                "surge_flag": temporal.surge_flag if temporal is not None else None,
                "bottleneck_pressure_60s": windows[60].bottleneck_pressure if 60 in windows else None,
                "net_flow_rate_60s": windows[60].net_flow_rate if 60 in windows else None,
                "window_maturity": {f"{s}s": {"sample_count": w.sample_count, "coverage_s": w.coverage_s, "is_mature": w.is_mature}
                                    for s, w in windows.items()},
            },
            "risk": predictions, "input_quality": quality,
        }

    def _pattern_rows(self, nan_idx: tuple[int, ...]) -> int | None:
        """Training rows (minimum over every loaded model) in which EXACTLY this set of model inputs was undefined.
        None if any model carries no training-pattern metadata (e.g. artifacts from before the reliability fix)."""
        support = []
        for model in self.risk_models.values():
            patterns = (getattr(model, "metadata", None) or {}).get("training_nan_patterns")
            if patterns is None:
                return None
            support.append(next((p["training_rows"] for p in patterns if tuple(p["nan_feature_indices"]) == nan_idx), 0))
        return min(support)

    def _input_quality(self, X: pd.DataFrame, zone_metrics: dict) -> dict[str, dict]:
        """Per zone: WHICH model inputs are undefined, WHY, and whether the models were trained on that exact state.
        Informational + a reliability gate for K1 only: it never changes a prediction and never imputes a value.
        Distinct causes are kept distinct: no_people | no_movement_observed (valid 0.0 speed, direction undefined) |
        movement_unavailable (people present but no usable track history) | warmup_history (only window statistics)."""
        names = self.risk_models["current"].feature_names
        min_rows = self.cfg.inference.min_pattern_support_rows
        out: dict[str, dict] = {}
        for zid in X.index:
            row = X.loc[zid]
            nan_idx = tuple(i for i, n in enumerate(names) if n not in X.columns or pd.isna(row[n]))
            m = zone_metrics[zid]
            if not nan_idx:
                state = "none"
            elif m.people_count == 0:
                state = "no_people"
            elif m.avg_speed_px_s is None:
                state = "movement_unavailable"
            elif m.direction_variance is None:
                state = "no_movement_observed"
            else:
                state = "warmup_history"
            rows = 0 if not nan_idx else self._pattern_rows(nan_idx)
            supported = not nan_idx or (rows is not None and rows >= min_rows)
            out[zid] = {
                "complete": not nan_idx, "missing_feature_count": len(nan_idx), "model_feature_count": len(names),
                "unavailable_metrics": [k for k in ("avg_speed_px_s", "speed_variance", "flow_direction_deg", "direction_variance") if getattr(m, k) is None],
                "undefined_state": state, "pattern_training_rows": rows if nan_idx else None,
                "pattern_supported": bool(supported), "reliable": bool(supported),
            }
        return out

    @staticmethod
    def _quality_note(q: dict) -> str | None:
        if q["complete"]:
            return None
        seen = "pattern never seen in training or no pattern metadata" if not q["pattern_supported"] else f"pattern seen in {q['pattern_training_rows']} training rows"
        return f"{q['undefined_state']}: {q['missing_feature_count']}/{q['model_feature_count']} model inputs undefined; {seen}"

    def _capabilities(self) -> dict:
        return {"future_5m": "future_5m" in self.risk_models, "future_10m": "future_10m" in self.risk_models,
                "explanations": bool(self.explainers) and self.cfg.inference.explain_mode != "off"}

    def _assemble(self, frame_id, ts, status, zones, explanations, recommendations, warnings, incomplete=(), unsupported=()) -> dict:
        return to_json_safe({
            "schema_version": SCHEMA_VERSION, "timestamp_s": ts, "frame_id": int(frame_id), "pipeline_status": status,
            "model_data_notice": MODEL_DATA_NOTICE, "capabilities": self._capabilities(), "zones": zones,
            "explanations": explanations, "diversion_recommendations": recommendations, "warnings": warnings,
            "incomplete_input_zones": list(incomplete), "unsupported_input_zones": list(unsupported),
        })

    def _no_data_result(self, frame_id: int, ts: float | None, warnings: list[str]) -> dict:
        zones = [self._zone_entry(z, None, None, None, None) for z in self._zone_configs]
        return self._assemble(frame_id, ts, "degraded", zones, {}, [], warnings)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_risk_models(model_dir: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        bundle = MultiHorizonRiskModel.load(model_dir)
        return {"current": bundle.current, "future_5m": bundle.future_5m, "future_10m": bundle.future_10m}, []
    except FileNotFoundError:
        try:
            current = RiskModel.load(model_dir / "risk_model.pkl")
        except FileNotFoundError as exc:
            raise PipelineError(f"no risk model artifacts found in {model_dir}") from exc
        except Exception as exc:
            raise PipelineError(f"failed to load H1 risk model from {model_dir}: {type(exc).__name__}: {exc}") from exc
        return {"current": current}, [f"I1 future-risk models not found in {model_dir}; running with the H1 current-risk model only - +5m/+10m unavailable"]
    except Exception as exc:
        raise PipelineError(f"failed to load risk models from {model_dir}: {type(exc).__name__}: {exc}") from exc


def _build_explainers(risk_models: dict[str, Any], cfg: AppConfig, root: Path) -> tuple[dict[str, ModelExplainer], list[str]]:
    if cfg.inference.explain_mode == "off":
        return {}, []
    try:
        if not cfg.inference.reference_dataset:
            raise FileNotFoundError("no reference_dataset configured")
        path = _resolve(root, cfg.inference.reference_dataset)
        if not path.exists():
            raise FileNotFoundError(f"reference dataset not found: {path}")
        names = risk_models["current"].feature_names
        wanted = set(names) | {"split", "label"}
        df = pd.read_csv(path, usecols=lambda c: c in wanted)
        if "split" in df.columns:
            df = df[df["split"] == "train"]
        X_ref, _, _ = prepare_features(df, feature_columns=names)
        return {key: ModelExplainer.from_training_data(m, X_ref, horizon=_HORIZON_NAME[key], horizon_seconds=HORIZON_SECONDS[_HORIZON_NAME[key]])
                for key, m in risk_models.items()}, []
    except Exception as exc:
        return {}, [f"explanations disabled: {type(exc).__name__}: {exc}"]
