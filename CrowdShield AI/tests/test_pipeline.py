"""L1 pipeline integration tests.

No YOLO weights, network, or video decoding are needed for ordinary tests: detectors are fakes
that return real `Detection` objects, the REAL ByteTrack tracker / zone / metric / temporal / K1 code
runs, and risk models are deterministic fakes (the packaged H1/I1 models are used in the
`real_models` tests only). The CLI is exercised on a tiny synthetic video written with OpenCV.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from config.loader import load_config
from config.schema import AppConfig, InferenceConfig
from decision.diversion_engine import DiversionEngine
from inference.pipeline import InferencePipeline, PipelineError, _load_risk_models, build_feature_row
from inference.schema import HORIZON_KEYS, to_json_safe, validate_result
from models.explainability import ModelExplainer
from models.features import LABELS_ORDERED
from models.future_risk import MultiHorizonRiskModel
from vision.benchmark import SampledFrame
from vision.detector import Detection
from vision.tracker import PersonTracker

REPO = Path(__file__).parent.parent
VENUE = REPO / "config" / "venues" / "k1_demo_venue.yaml"
FRAME = np.zeros((220, 340, 3), dtype=np.uint8)
# zone anchors (x, y) chosen inside the k1_demo_venue polygons
ANCHORS = {"entrance": (20, 50), "main_hall": (120, 50), "exit_north": (230, 50), "exit_south": (120, 160)}


# ------------------------------------------------------------------ fakes/helpers
class FakeDetector:
    def __init__(self, scene=None, fail_on=()):
        self.scene, self.fail_on, self.calls = scene or {}, set(fail_on), 0

    def detect(self, frame):
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("detector exploded")
        return [Detection(ANCHORS[z][0] + 12 * i - 4, ANCHORS[z][1] - 15, ANCHORS[z][0] + 12 * i + 4, ANCHORS[z][1] + 15, 0.9)
                for z, n in self.scene.items() for i in range(n)]


class FakeModel:
    """Deterministic stand-in with the RiskModel surface the pipeline/explainer use."""

    def __init__(self, by_zone=None, rule=None, feature_names=("people_count", "density"), fail=False, bad=None):
        self.by_zone, self.rule, self.feature_names = by_zone or {}, rule, list(feature_names)
        self.model_version, self.fail, self.bad, self.calls = "fake-1", fail, bad, []

    def predict_proba(self, X):
        self.calls.append(X.copy())
        if self.fail:
            raise RuntimeError("model exploded")
        labels = self.rule(X) if self.rule else [self.by_zone.get(z, "SAFE") for z in X.index]
        P = pd.DataFrame(0.0, index=X.index, columns=LABELS_ORDERED)
        for zid, label in zip(X.index, labels):
            P.loc[zid, label] = 1.0
        if self.bad == "nan_first_row":
            P.iloc[0, :] = np.nan
        if self.bad == "columns":
            P = P.rename(columns={"SAFE": "OK"})
        return P


def cfg_with(cfg, **inference):
    return AppConfig.model_validate({**cfg.model_dump(), "inference": {**cfg.inference.model_dump(), **inference}})


def make_pipeline(cfg, scene=None, models=None, explainers=None, detector=None, tracker=None):
    models = models if models is not None else {k: FakeModel() for k in HORIZON_KEYS}
    return InferencePipeline(cfg, detector or FakeDetector(scene), tracker or PersonTracker(30, 15.0), models, explainers or {})


def run_frames(pipe, n, step=1.0):
    results = [pipe.process_frame(FRAME, i, i * step) for i in range(n)]
    return results[-1], results


def zone(result, zid):
    return next(z for z in result["zones"] if z["zone_id"] == zid)


class FakeSource:
    def __init__(self, n):
        self.n, self.closed = n, False

    def __iter__(self):
        return self._gen()

    def _gen(self):
        try:
            for i in range(self.n):
                yield SampledFrame(FRAME, i, float(i))
        finally:
            self.closed = True


@pytest.fixture(scope="module")
def cfg():
    return load_config(venue_path=VENUE)


@pytest.fixture(scope="module")
def valid_result(cfg):
    return run_frames(make_pipeline(cfg, {"entrance": 2, "main_hall": 3}), 3)[0]


# ------------------------------------------------------------- A. mocked full path
def test_complete_mocked_path_produces_valid_structured_output(cfg):
    models = {k: FakeModel(by_zone={"main_hall": "HIGH"}) for k in HORIZON_KEYS}
    pipe = make_pipeline(cfg, {"entrance": 3, "main_hall": 2}, models)
    result, _ = run_frames(pipe, 3)
    assert validate_result(result) == []
    assert result["frame_id"] == 2 and result["timestamp_s"] == 2.0
    assert [z["zone_id"] for z in result["zones"]] == [z.id for z in cfg.venue.zones]
    assert zone(result, "entrance")["people_count"] == 3 and zone(result, "main_hall")["people_count"] == 2
    assert zone(result, "main_hall")["risk"]["current"]["risk_class"] == "HIGH"
    assert zone(result, "entrance")["risk"]["future_10m"]["risk_class"] == "SAFE"
    assert zone(result, "entrance")["utilization"] == pytest.approx(3 / 80)
    assert "synthetic" in result["model_data_notice"].lower()
    assert [r["source_zone"] for r in result["diversion_recommendations"]] == ["main_hall"]
    assert result["diversion_recommendations"][0]["status"] == "RECOMMENDED"


def test_data_flow_into_models_and_k1(cfg):
    model = FakeModel(by_zone={"main_hall": "HIGH"})
    pipe = make_pipeline(cfg, {"entrance": 3, "main_hall": 2}, {k: model for k in HORIZON_KEYS})
    captured = []
    original = pipe.engine.recommend_all
    pipe.engine.recommend_all = lambda states, ts: captured.append(states) or original(states, ts)
    result, _ = run_frames(pipe, 2)
    X = model.calls[-1]
    assert list(X.index) == ["entrance", "main_hall", "exit_north", "exit_south", "overflow_zone"]  # disabled zone excluded
    assert X["people_count"].tolist() == [3, 2, 0, 0, 0] and "w60_density_max" in X.columns and "surge_flag" in X.columns
    states = captured[-1]
    assert states["main_hall"].current_risk_label == "HIGH" and states["main_hall"].people_count == 2
    assert states["entrance"].capacity == 80 and "maintenance_closet" not in states


def test_recommendations_are_computed_by_k1_not_hardcoded(cfg):
    def last(by_zone):
        pipe = make_pipeline(cfg, {"main_hall": 2}, {k: FakeModel(by_zone=by_zone) for k in HORIZON_KEYS})
        spy = {}
        original = pipe.engine.recommend_all
        pipe.engine.recommend_all = lambda states, ts: spy.setdefault("states", states) and original(states, ts)
        result, _ = run_frames(pipe, 2)
        return result, spy["states"]

    result, states = last({"main_hall": "HIGH"})
    rec = result["diversion_recommendations"][0]
    independent = DiversionEngine(pipe_graph := make_pipeline(cfg).graph, cfg.diversion).recommend_all(states, 1.0)["main_hall"]
    assert rec["destination_zone"] == independent.destination_zone and rec["reason"] == independent.reason
    # change only the model outputs -> the recommendation must change accordingly
    blocked, _ = last({"main_hall": "HIGH", "exit_north": "CRITICAL"})
    brec = next(r for r in blocked["diversion_recommendations"] if r["source_zone"] == "main_hall")
    assert brec["destination_zone"] != "exit_north"
    assert "exit_north" in {c["zone_id"] for c in brec["rejected_candidates"]}
    nothing, _ = last({"main_hall": "HIGH", "entrance": "CRITICAL", "exit_north": "CRITICAL", "exit_south": "CRITICAL"})
    nrec = next(r for r in nothing["diversion_recommendations"] if r["source_zone"] == "main_hall")
    assert nrec["status"] == "NO_SAFE_DESTINATION" and nrec["destination_zone"] is None and nrec["rejected_candidates"]


# --------------------------------------------------------------------- JSON safety
def test_to_json_safe_handles_numpy_nan_inf_and_dataclasses():
    from decision.schemas import RejectedCandidate

    raw = {"a": np.float64("nan"), "b": np.int64(3), "c": np.array([1.5, np.inf]), "d": np.bool_(True), "e": np.float32(2.5),
           "f": RejectedCandidate("z", "why"), "g": (1, np.int32(2)), "h": float("-inf"), "zero": 0, "zerof": np.float64(0.0)}
    safe = to_json_safe(raw)
    text = json.dumps(safe, allow_nan=False)
    assert safe["a"] is None and safe["c"] == [1.5, None] and safe["h"] is None
    assert safe["b"] == 3 and safe["d"] is True and safe["f"] == {"zone_id": "z", "reason": "why"}
    assert safe["zero"] == 0 and safe["zerof"] == 0.0 and json.loads(text)["zerof"] == 0.0  # real zero stays zero


def test_result_is_strict_json_with_no_nan_or_infinity(valid_result):
    text = json.dumps(valid_result, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == valid_result


def test_unavailable_is_null_and_real_zero_is_zero(cfg):
    result, _ = run_frames(make_pipeline(cfg, scene={}), 3)  # empty scene: nobody present
    z = zone(result, "main_hall")
    assert z["people_count"] == 0 and z["density"] == 0.0 and z["utilization"] == 0.0  # genuine zeros
    assert z["metrics"]["avg_speed_px_s"] is None  # undefined with nobody moving -> null, not 0


# ------------------------------------------------- empty / disabled / warm-up states
def test_empty_scene_is_valid_and_not_a_failure(cfg):
    result, _ = run_frames(make_pipeline(cfg, scene={}), 3)
    assert validate_result(result) == [] and result["pipeline_status"] == "warming_up"
    assert result["diversion_recommendations"] == []
    assert zone(result, "entrance")["risk"]["current"]["status"] == "ok"  # a genuine prediction, not unavailable


def test_disabled_zone_is_reported_but_excluded(cfg):
    result, _ = run_frames(make_pipeline(cfg, {"entrance": 1}), 2)
    z = zone(result, "maintenance_closet")
    assert z["status"] == "disabled" and z["people_count"] is None and z["risk"] is None and z["input_quality"] is None
    assert any("maintenance_closet" in w and "disabled" in w for w in result["warnings"])


def test_temporal_warmup_then_mature(cfg):
    pipe = make_pipeline(cfg, {"entrance": 1})
    _, results = run_frames(pipe, 64)
    first, last = results[0], results[-1]
    assert first["pipeline_status"] == "warming_up" and zone(first, "entrance")["status"] == "warmup"
    assert zone(first, "entrance")["temporal"]["window_maturity"]["60s"]["is_mature"] is False
    assert last["pipeline_status"] == "ok" and zone(last, "entrance")["status"] == "ok"
    assert zone(last, "entrance")["temporal"]["window_maturity"]["60s"]["is_mature"] is True


# ------------------------------------------ unavailable / invalid model behaviour
def test_missing_future_models_are_unavailable_not_safe(cfg):
    pipe = make_pipeline(cfg, {"main_hall": 2}, {"current": FakeModel(by_zone={"main_hall": "HIGH"})})
    result, _ = run_frames(pipe, 2)
    risk = zone(result, "main_hall")["risk"]
    assert risk["current"]["status"] == "ok"
    for key in ("future_5m", "future_10m"):
        assert risk[key]["status"] == "unavailable" and risk[key]["risk_class"] is None and risk[key]["reason"] == "model not loaded"
    assert result["pipeline_status"] == "degraded" and result["capabilities"]["future_5m"] is False
    assert validate_result(result) == []
    assert result["diversion_recommendations"][0]["model_horizon_used"] == "current"


@pytest.mark.parametrize("bad", ["raise", "nan_first_row", "columns"])
def test_invalid_model_output_is_unavailable_and_excluded_from_k1(cfg, bad):
    broken = FakeModel(fail=True) if bad == "raise" else FakeModel(by_zone={"entrance": "HIGH"}, bad=bad)
    models = {"current": broken, "future_5m": FakeModel(), "future_10m": FakeModel()}
    pipe = make_pipeline(cfg, {"entrance": 2, "main_hall": 1}, models)
    seen = {}
    original = pipe.engine.recommend_all
    pipe.engine.recommend_all = lambda states, ts: seen.update(states) or original(states, ts)
    result, _ = run_frames(pipe, 2)
    entrance = zone(result, "entrance")
    assert entrance["status"] == "unavailable" and entrance["risk"]["current"]["risk_class"] is None
    assert entrance["risk"]["current"]["risk_score"] is None and "unavailable" in " ".join(result["warnings"]).lower() or bad == "raise"
    assert "entrance" not in seen  # never handed to K1 as if it were safe
    assert result["pipeline_status"] == "degraded" and validate_result(result) == []


# ------------------------------------------------- component / input failures
def test_detector_failure_degrades_one_frame_then_recovers(cfg):
    pipe = make_pipeline(cfg, detector=FakeDetector({"entrance": 1}, fail_on={2}))
    r1 = pipe.process_frame(FRAME, 0, 0.0)
    r2 = pipe.process_frame(FRAME, 1, 1.0)
    r3 = pipe.process_frame(FRAME, 2, 2.0)
    assert r1["pipeline_status"] != "degraded" and r3["pipeline_status"] != "degraded"
    assert r2["pipeline_status"] == "degraded" and r2["diversion_recommendations"] == []
    assert zone(r2, "entrance")["status"] == "no_data" and zone(r2, "entrance")["people_count"] is None
    assert any("detection/tracking failed" in w for w in r2["warnings"]) and validate_result(r2) == []


def test_repeated_detector_failure_aborts_with_clear_error(cfg):
    pipe = make_pipeline(cfg_with(cfg, max_consecutive_failures=2), detector=FakeDetector(fail_on={1, 2, 3}))
    assert pipe.process_frame(FRAME, 0, 0.0)["pipeline_status"] == "degraded"
    with pytest.raises(PipelineError, match="consecutive failures"):
        pipe.process_frame(FRAME, 1, 1.0)


@pytest.mark.parametrize("frame", [None, np.zeros((0, 0, 3), np.uint8), np.zeros((10, 10), np.uint8), "frame"])
def test_invalid_frame_is_skipped_without_calling_detector(cfg, frame):
    detector = FakeDetector({"entrance": 1})
    result = make_pipeline(cfg, detector=detector).process_frame(frame, 0, 0.0)
    assert detector.calls == 0 and result["pipeline_status"] == "degraded"
    assert any("invalid frame" in w for w in result["warnings"]) and validate_result(result) == []


def test_non_monotonic_or_invalid_timestamp_is_rejected(cfg):
    pipe = make_pipeline(cfg, {"entrance": 1})
    pipe.process_frame(FRAME, 0, 5.0)
    back = pipe.process_frame(FRAME, 1, 4.0)
    nan = pipe.process_frame(FRAME, 2, float("nan"))
    assert back["pipeline_status"] == "degraded" and any("precedes" in w for w in back["warnings"])
    assert nan["timestamp_s"] is None and any("invalid timestamp" in w for w in nan["warnings"])
    assert pipe.process_frame(FRAME, 3, 5.0)["pipeline_status"] != "degraded"  # state was not corrupted


def test_incomplete_inputs_are_flagged_not_hidden(cfg):
    model = FakeModel(feature_names=("people_count", "density", "avg_speed_px_s"))
    pipe = make_pipeline(cfg, {}, {k: model for k in HORIZON_KEYS})
    result, _ = run_frames(pipe, 2)
    assert "entrance" in result["incomplete_input_zones"]
    q = zone(result, "entrance")["input_quality"]
    assert q["complete"] is False and "avg_speed_px_s" in q["unavailable_metrics"] and q["missing_feature_count"] == 1
    assert any("LOW-RELIABILITY" in w for w in result["warnings"])
    # a zone's prediction is still reported as computed (not suppressed / imputed)
    assert zone(result, "entrance")["risk"]["current"]["status"] == "ok"


def test_recommendation_carries_input_completeness(cfg):
    # two STATIONARY people: speed is a defined 0.0, but movement direction is undefined (None -> NaN)
    model = FakeModel(by_zone={"main_hall": "HIGH"}, feature_names=("people_count", "avg_speed_px_s", "direction_variance"))
    result, _ = run_frames(make_pipeline(cfg, {"main_hall": 2}, {k: model for k in HORIZON_KEYS}), 2)
    q = zone(result, "main_hall")["input_quality"]
    assert q["unavailable_metrics"] == ["flow_direction_deg", "direction_variance"] and q["missing_feature_count"] == 1
    assert zone(result, "main_hall")["metrics"]["avg_speed_px_s"] == 0.0  # genuine zero, not null
    assert result["diversion_recommendations"][0]["source_inputs_complete"] is False


# --------------------------------------------------------------- explanations
def _explainers(model):
    return {k: ModelExplainer(model, pd.Series({"people_count": 0.0, "density": 0.0}), horizon=k) for k in HORIZON_KEYS}


def test_explanations_ground_in_model_features_and_respect_modes(cfg):
    rule = lambda X: ["HIGH" if v >= 3 else "SAFE" for v in X["people_count"]]
    model = FakeModel(rule=rule)
    models = {k: model for k in HORIZON_KEYS}
    result, _ = run_frames(make_pipeline(cfg_with(cfg, explain_mode="alerts"), {"entrance": 3, "main_hall": 1}, models, _explainers(model)), 2)
    assert set(result["explanations"]) == {"entrance"}  # only the HIGH zone is explained in 'alerts' mode
    ex = result["explanations"]["entrance"]["current"]
    assert ex["status"] == "ok" and ex["predicted_class"] == "HIGH"
    assert ex["top_positive_features"][0]["feature"] in ("people_count", "density")
    assert ex["top_positive_features"][0]["direction"] == "supports_prediction"
    assert validate_result(result) == []
    every, _ = run_frames(make_pipeline(cfg_with(cfg, explain_mode="all"), {"entrance": 3}, models, _explainers(model)), 1)
    assert len(every["explanations"]) == 5
    off, _ = run_frames(make_pipeline(cfg_with(cfg, explain_mode="off"), {"entrance": 3}, models, _explainers(model)), 1)
    assert off["explanations"] == {} and off["capabilities"]["explanations"] is False


def test_explainer_failure_is_isolated(cfg):
    class Broken:
        def explain(self, *a, **k):
            raise ValueError("explainer exploded")

    model = FakeModel(by_zone={"entrance": "HIGH"})
    pipe = make_pipeline(cfg_with(cfg, explain_mode="all"), {"entrance": 2}, {k: model for k in HORIZON_KEYS}, {k: Broken() for k in HORIZON_KEYS})
    result, _ = run_frames(pipe, 1)
    assert result["explanations"]["entrance"]["current"]["status"] == "unavailable"
    assert any("explanation unavailable" in w for w in result["warnings"])
    assert zone(result, "entrance")["risk"]["current"]["risk_class"] == "HIGH" and validate_result(result) == []


# ------------------------------------------------- lifecycle / memory / cleanup
def test_source_closed_on_completion_early_stop_and_error(cfg):
    pipe = make_pipeline(cfg, {"entrance": 1})
    done = FakeSource(4)
    assert len(list(pipe.run(done))) == 4 and done.closed
    early = FakeSource(10)
    assert len(list(pipe.run(early, max_frames=3))) == 3 and early.closed
    zero = FakeSource(10)
    assert list(pipe.run(zero, max_frames=0)) == [] and zero.closed
    failing = FakeSource(10)
    broken = make_pipeline(cfg_with(cfg, max_consecutive_failures=1), detector=FakeDetector(fail_on={1}))
    with pytest.raises(PipelineError):
        list(broken.run(failing))
    assert failing.closed
    with pytest.raises(ValueError):
        list(pipe.run(FakeSource(1), max_frames=-1))


def _write_video(path, n=8, fps=12):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        writer.write(np.full((48, 64, 3), i * 10, np.uint8))
    writer.release()


def test_real_framesource_is_released_in_all_exit_paths(cfg, tmp_path, monkeypatch):
    from vision.benchmark import FrameSource

    video = tmp_path / "v.mp4"
    _write_video(video)
    counts = {"open": 0, "release": 0}
    real = cv2.VideoCapture

    class Proxy:
        def __init__(self, *a):
            self._c = real(*a)
            counts["open"] += 1

        def release(self):
            counts["release"] += 1
            self._c.release()

        def __getattr__(self, name):
            return getattr(self._c, name)

    monkeypatch.setattr(cv2, "VideoCapture", Proxy)
    pipe = make_pipeline(cfg, {"entrance": 1})
    assert len(list(pipe.run(FrameSource(str(video), 12.0)))) == 8
    assert counts["open"] == counts["release"] > 0
    list(pipe.run(FrameSource(str(video), 12.0), max_frames=2))
    assert counts["open"] == counts["release"]
    broken = make_pipeline(cfg_with(cfg, max_consecutive_failures=1), detector=FakeDetector(fail_on={1}))
    with pytest.raises(PipelineError):
        list(broken.run(FrameSource(str(video), 12.0)))
    assert counts["open"] == counts["release"]


def test_long_run_keeps_state_bounded(cfg):
    pipe = make_pipeline(cfg, {"entrance": 2, "main_hall": 2})
    for i in range(700):
        result = pipe.process_frame(FRAME, i, float(i))
    assert all(len(t._history) <= 62 for t in pipe.temporal_builder._trackers.values())  # 60s window at 1 Hz
    assert len(pipe.metrics_calculator.transition_tracker._last_zone) <= pipe.metrics_calculator.transition_tracker.max_tracked_ids
    assert len(pipe.tracker._trajectories) <= 10 and len(pipe._last_classes) == 5
    assert validate_result(result) == []


# ----------------------------------------------------------------- loading/config
def test_feature_row_matches_real_model_schema_and_encoding():
    bundle = MultiHorizonRiskModel.load(REPO / "models" / "artifacts")
    from features.crowd_metrics import ZoneMetrics
    from features.temporal import TemporalFeatureBuilder

    metrics = ZoneMetrics("z", 1.0, 3, 0.1, None, None, None, None, 0, 0)
    temporal = TemporalFeatureBuilder(["z"], [5, 15, 30, 60], 0.05, 60.0).update({"z": metrics})["z"]
    row = build_feature_row(metrics, temporal)
    assert set(bundle.current.feature_names) <= set(row)
    assert all(isinstance(v, float) for v in row.values())
    assert row["surge_flag"] == 0.0 and np.isnan(row["avg_speed_px_s"]) and row["people_count"] == 3.0
    assert row["w5_is_mature"] in (0.0, 1.0)


def test_model_loading_failures_are_clear(tmp_path, cfg, monkeypatch):
    with pytest.raises(PipelineError, match="no risk model artifacts"):
        _load_risk_models(tmp_path)
    shutil.copy(REPO / "models/artifacts/risk_model.pkl", tmp_path / "risk_model.pkl")
    shutil.copy(REPO / "models/artifacts/risk_model.pkl.meta.json", tmp_path / "risk_model.pkl.meta.json")
    models, warnings = _load_risk_models(tmp_path)
    assert set(models) == {"current"} and "H1" in warnings[0]
    import vision.detector

    class Boom:
        def __init__(self, **kw):
            raise RuntimeError("no weights")

    monkeypatch.setattr(vision.detector, "PersonDetector", Boom)
    with pytest.raises(PipelineError, match="person detector"):
        InferencePipeline.from_config(cfg)
    with pytest.raises(PipelineError, match="no enabled zones"):
        InferencePipeline(load_config(), FakeDetector(), PersonTracker(30, 15.0), {"current": FakeModel()})


def test_inference_config_defaults_and_validation():
    assert InferenceConfig().explain_mode == "alerts" and load_config().inference.model_dir == "models/artifacts"
    with pytest.raises(Exception):
        InferenceConfig(explain_mode="sometimes")
    with pytest.raises(Exception):
        InferenceConfig(max_consecutive_failures=0)


def test_requirements_declare_lap_for_reproducible_tracking():
    lines = [l.strip().lower() for l in (REPO / "requirements.txt").read_text().splitlines()]
    assert any(l.startswith("lap") for l in lines)


# ----------------------------------------------------------------- schema validator
@pytest.mark.parametrize("mutate,expect", [
    (lambda r: r.pop("warnings"), "missing top-level key"),
    (lambda r: r.update(pipeline_status="great"), "pipeline_status invalid"),
    (lambda r: r["zones"][0].update(status="bogus"), "status invalid"),
    (lambda r: r["zones"][0]["risk"]["current"].update(risk_score=150.0), "risk_score"),
    (lambda r: r["zones"][0]["risk"]["current"].update(risk_class="MEH"), "risk_class"),
    (lambda r: r["zones"][0].update(density=float("nan")), "non-finite"),
    (lambda r: r["zones"][-1].update(people_count=0), "disabled zones must report people_count null"),
    (lambda r: r.update(frame_id="3"), "frame_id"),
])
def test_validator_rejects_violations(valid_result, mutate, expect):
    bad = copy.deepcopy(valid_result)
    mutate(bad)
    assert any(expect in e for e in validate_result(bad)), validate_result(bad)


def test_validator_recommendation_consistency(valid_result):
    bad = copy.deepcopy(valid_result)
    bad["diversion_recommendations"] = [{"source_zone": "a", "destination_zone": None, "status": "RECOMMENDED", "urgency": "NORMAL",
                                         "reason": "x", "rejected_candidates": [], "source_inputs_complete": True}]
    assert any("destination_zone must be set iff" in e for e in validate_result(bad))
    assert validate_result({"nope": 1}) and validate_result("not a dict") == ["result must be an object"]


# ------------------------------------------------------------------------- CLI
@pytest.fixture
def isolated_logging():
    """run_cli() calls setup_logging(), which mutates the process-global 'crowdshield' logger (level, handlers, streams).
    Restore it so later tests (e.g. test_smoke's DEBUG-level check, which assumes first-call configuration) are unaffected."""
    import logging

    logger = logging.getLogger("crowdshield")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    streams = [(h, h.stream) for h in handlers if hasattr(h, "stream")]
    yield logger
    for h in list(logger.handlers):
        if h not in handlers:
            logger.removeHandler(h)
    for h in handlers:
        if h not in logger.handlers:
            logger.addHandler(h)
    for h, stream in streams:
        h.setStream(stream)
    logger.setLevel(level)
    logger.propagate = propagate


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_pipeline_cli", REPO / "scripts" / "run_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_help_and_invalid_paths_via_subprocess(tmp_path):
    script = str(REPO / "scripts" / "run_pipeline.py")
    helped = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True, cwd=REPO)
    assert helped.returncode == 0 and "--max-frames" in helped.stdout and "--validate" in helped.stdout
    missing = subprocess.run([sys.executable, script, "--video", str(tmp_path / "nope.mp4")], capture_output=True, text=True, cwd=REPO)
    assert missing.returncode == 2 and "video not found" in missing.stderr
    video = tmp_path / "v.mp4"
    _write_video(video, 2)
    badcfg = subprocess.run([sys.executable, script, "--video", str(video), "--venue", str(tmp_path / "x.yaml")], capture_output=True, text=True, cwd=REPO)
    assert badcfg.returncode == 2 and "invalid configuration" in badcfg.stderr
    nomodel = subprocess.run([sys.executable, script, "--video", str(video), "--model-dir", str(tmp_path)], capture_output=True, text=True, cwd=REPO)
    assert nomodel.returncode != 0 and ("error:" in nomodel.stderr)


def test_cli_end_to_end_with_injected_pipeline_writes_valid_jsonl(cfg, tmp_path, isolated_logging):
    cli = _load_cli()
    video, out = tmp_path / "v.mp4", tmp_path / "out" / "r.jsonl"
    _write_video(video, 8)
    factory = lambda c: make_pipeline(c, {"entrance": 2, "main_hall": 1}, {k: FakeModel(by_zone={"main_hall": "HIGH"}) for k in HORIZON_KEYS})
    code = cli.run_cli(["--video", str(video), "--venue", str(VENUE), "--max-frames", "5", "--output", str(out), "--validate"], pipeline_factory=factory)
    assert code == 0
    lines = out.read_text().splitlines()
    assert len(lines) == 5
    results = [json.loads(l) for l in lines]
    assert all(validate_result(r) == [] for r in results) and [r["frame_id"] for r in results] == [0, 1, 2, 3, 4]
    assert results[-1]["diversion_recommendations"][0]["source_zone"] == "main_hall"
    assert cli.run_cli(["--video", str(video), "--max-frames", "-1"], pipeline_factory=factory) == 2


def test_cli_stdout_mode_keeps_logs_off_stdout(cfg, tmp_path, capsys, isolated_logging):
    cli = _load_cli()
    video = tmp_path / "v.mp4"
    _write_video(video, 3)
    code = cli.run_cli(["--video", str(video), "--venue", str(VENUE), "--max-frames", "2", "--output", "-"],
                       pipeline_factory=lambda c: make_pipeline(c, {"entrance": 1}))
    captured = capsys.readouterr()
    assert code == 0
    assert [json.loads(l)["frame_id"] for l in captured.out.splitlines()] == [0, 1]  # stdout is pure JSONL
    assert "processed 2 frame(s)" in captured.err


def test_cli_run_does_not_leak_logging_state(tmp_path, isolated_logging):
    """Regression: an in-process CLI run must not be able to change what a later setup_logging('DEBUG') does."""
    import logging

    logger = logging.getLogger("crowdshield")
    for h in list(logger.handlers):
        logger.removeHandler(h)  # simulate a pristine process
    logger.setLevel(logging.NOTSET)
    video = tmp_path / "v.mp4"
    _write_video(video, 2)
    _load_cli().run_cli(["--video", str(video), "--venue", str(VENUE), "--max-frames", "1", "--output", str(tmp_path / "o.jsonl")],
                        pipeline_factory=lambda c: make_pipeline(c, {"entrance": 1}))
    assert logger.level == logging.INFO  # the CLI configured it (expected inside the test) ...
    # ... and the fixture (verified by the next test in the full suite, test_smoke) restores the original state on teardown.


# ------------------------------------------------ real packaged models (no YOLO)
needs_artifacts = pytest.mark.skipif(
    not (REPO / "models/artifacts/future_risk_5m.pkl").exists() or not (REPO / "sample_data/synthetic_long/synthetic_dataset.csv").exists(),
    reason="packaged I1 models / reference dataset not present",
)


@needs_artifacts
def test_real_models_end_to_end_with_fake_detector(cfg):
    pipe = InferencePipeline.from_config(cfg_with(cfg, explain_mode="all"), detector=FakeDetector({"entrance": 3, "main_hall": 2}),
                                         tracker=PersonTracker(30, 15.0))
    assert pipe.risk_models.keys() == {"current", "future_5m", "future_10m"} and pipe.explainers
    result, _ = run_frames(pipe, 3)
    assert validate_result(result) == [] and json.dumps(result, allow_nan=False)
    assert result["capabilities"] == {"future_5m": True, "future_10m": True, "explanations": True}
    for z in result["zones"]:
        if z["enabled"]:
            assert all(z["risk"][k]["status"] == "ok" for k in HORIZON_KEYS), z["zone_id"]  # no feature-schema mismatch
            assert abs(sum(z["risk"]["current"]["class_probabilities"].values()) - 1.0) < 1e-6
    ex = result["explanations"]["main_hall"]["current"]
    names = set(pipe.risk_models["current"].feature_names)
    assert ex["status"] == "ok" and {f["feature"] for f in ex["top_positive_features"] + ex["top_negative_features"]} <= names


@needs_artifacts
def test_real_models_flag_empty_zones_as_incomplete_inputs(cfg):
    """Pins the L1 finding AND its resolution. Undefined speed/direction inputs (nobody moving) were absent from the original synthetic
    training data, so predictions there were out-of-distribution. After the reliability fix (data/scenarios_v2.yaml) the SAME inputs are
    still reported as undefined (incomplete) - undefined is never hidden or imputed - but the retrained models were trained on exactly
    that state, so the zone is SUPPORTED (not out-of-distribution) and raises no false alarm."""
    pipe = InferencePipeline.from_config(cfg_with(cfg, explain_mode="off"), detector=FakeDetector({}), tracker=PersonTracker(30, 15.0))
    result, _ = run_frames(pipe, 70)
    assert set(result["incomplete_input_zones"]) >= {"entrance", "main_hall"}
    assert zone(result, "main_hall")["input_quality"]["missing_feature_count"] > 0
    assert all(r["source_inputs_complete"] is False for r in result["diversion_recommendations"])
    assert result["unsupported_input_zones"] == [] and result["diversion_recommendations"] == []  # supported state, no false EMERGENCY
    assert zone(result, "main_hall")["input_quality"]["undefined_state"] == "no_people"
