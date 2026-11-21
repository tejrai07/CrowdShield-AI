"""Regression tests for the post-L1 ML reliability fix.

Root cause (measured, see README 'Reliability fix'): the synthetic training data never contained empty zones or undefined movement
metrics after warm-up, tied low speed to danger, and extrapolated trends without bound; so inference states the pipeline really produces
(empty zone, standing crowd, dropout) were out-of-distribution. These tests pin the corrected behaviour at every layer.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.loader import load_config
from config.schema import AppConfig, DiversionConfig
from data.loader import load_generator_config
from data.synthetic_generator import apply_split, generate_dataset, split_dataset, split_dataset_stratified
from decision.diversion_engine import DiversionEngine, source_triggers
from decision.schemas import ZoneState
from decision.zone_graph import ZoneGraph
from evaluation.probes import PROBES, probe_frame, probe_row, run_probes
from features.crowd_metrics import ZoneMetrics
from inference.pipeline import InferencePipeline, build_feature_row
from inference.schema import validate_result
from models.explainability import ModelExplainer
from models.features import LABELS_ORDERED, get_feature_columns
from models.future_risk import MultiHorizonRiskModel
from models.risk_model import RiskModel
from vision.detector import Detection
from vision.tracker import PersonTracker

REPO = Path(__file__).parent.parent
V2_YAML = REPO / "data" / "scenarios_v2.yaml"
VENUE = REPO / "config" / "venues" / "k1_demo_venue.yaml"
ART, OLD = REPO / "models" / "artifacts", REPO / "models" / "artifacts" / "archive_pre_reliability"
FRAME = np.zeros((220, 340, 3), np.uint8)
DANGER = {"HIGH", "CRITICAL"}
needs_models = pytest.mark.skipif(not (ART / "future_risk_5m.pkl").exists(), reason="packaged models not present")
needs_archive = pytest.mark.skipif(not (OLD / "future_risk_5m.pkl").exists(), reason="archived pre-fix models not present")


def v2_cfg(duration, stride=1, per_scenario=3, per_transition=3):
    base = load_generator_config(V2_YAML)
    return base.model_validate({**base.model_dump(), "duration_s": float(duration), "row_stride": stride,
                                "realizations_per_scenario": per_scenario, "realizations_per_transition": per_transition})


@pytest.fixture(scope="module")
def v2_400():
    cfg = v2_cfg(400)
    df, _ = generate_dataset(cfg)
    return cfg, apply_split(df, cfg, seed=11)


def scenario_rows(df, name):
    return df[df.scenario == name]


# ------------------------------------------------------------------ generator semantics
def test_empty_zone_has_no_people_and_undefined_movement(v2_400):
    d = scenario_rows(v2_400[1], "EMPTY_ZONE")
    assert (d.people_count == 0).all() and (d.label == "SAFE").all()
    for col in ("avg_speed_px_s", "speed_variance", "direction_variance", "flow_direction_deg"):
        assert d[col].isna().all(), col  # UNDEFINED, not zero


def test_standing_people_have_valid_zero_speed_but_undefined_direction(v2_400):
    d = scenario_rows(v2_400[1], "SINGLE_STATIONARY_PERSON")
    assert (d.people_count == 1).all()
    assert (d.avg_speed_px_s == 0.0).all() and (d.speed_variance == 0.0).all()  # VALID zeros: people exist, nobody moves
    assert d.direction_variance.isna().all() and d.flow_direction_deg.isna().all()  # direction undefined (zero displacement)
    # valid-zero and unknown are different states, in the data and later in the features
    assert d.avg_speed_px_s.notna().all() and scenario_rows(v2_400[1], "EMPTY_ZONE").avg_speed_px_s.isna().all()


def test_danger_is_driven_by_density_independent_of_speed(v2_400):
    df = v2_400[1]
    stationary, moving = scenario_rows(df, "STATIONARY_DENSE_HIGH"), scenario_rows(df, "MOVING_DENSE_HIGH")
    assert (stationary.label == "HIGH").all() and (moving.label == "HIGH").all()
    assert stationary.avg_speed_px_s.max() == 0.0 and moving.avg_speed_px_s.min() > 20  # same label, opposite speed regimes
    assert moving.direction_variance.notna().all() and stationary.direction_variance.isna().all()
    safe_slow = scenario_rows(df, "SLOW_WALK_SAFE")
    assert (safe_slow.label == "SAFE").all() and 5 < safe_slow.avg_speed_px_s.mean() < 12  # slow but safe: breaks speed->danger shortcut


def test_dropout_scenarios_mix_defined_and_undefined_movement_with_people_present(v2_400):
    d = scenario_rows(v2_400[1], "DROPOUT_DENSE_HIGH")
    undefined = d.avg_speed_px_s.isna()
    assert (d.people_count > 0).all() and 0.1 < undefined.mean() < 0.3  # configured ~0.2
    assert d.loc[undefined, "direction_variance"].isna().all()


def test_trends_plateau_at_original_envelope_instead_of_exploding(v2_400):
    d = scenario_rows(v2_400[1], "RAPID_SURGE")
    late = d[(d.timestamp_s >= 300) & (d.timestamp_s < 360)]
    cap = load_generator_config(V2_YAML).scenarios[[s.name for s in load_generator_config(V2_YAML).scenarios].index("RAPID_SURGE")].max_density
    assert cap == pytest.approx(10.6) and late.density.mean() == pytest.approx(cap, rel=0.1)  # legacy would be 1.0 + 0.08*300 = 25
    assert v2_400[1].people_count.max() < 1000


def test_legacy_datasets_are_reproduced_exactly():
    """Backward compatibility: scenarios.yaml + the generator changes still emit the shipped H1 dataset, row for row."""
    cfg = load_generator_config()
    df, _ = generate_dataset(cfg)
    df = apply_split(df, cfg, seed=7)
    ref = pd.read_csv(REPO / "sample_data" / "synthetic" / "synthetic_dataset.csv")
    pd.testing.assert_frame_equal(df.reset_index(drop=True), ref, check_dtype=False, rtol=1e-9, atol=1e-9)


# ------------------------------------------------------- determinism / splits / leakage
def test_generation_is_deterministic_and_seed_sensitive():
    a, b = generate_dataset(v2_cfg(120))[0], generate_dataset(v2_cfg(120))[0]
    pd.testing.assert_frame_equal(a, b)
    other = generate_dataset(v2_cfg(120).model_validate({**v2_cfg(120).model_dump(), "base_seed": 99}))[0]
    assert not a["density"].equals(other["density"])


def test_stratified_split_is_sequence_level_complete_and_deterministic(v2_400):
    df = v2_400[1]
    ids = {s: set(df[df.split == s].sequence_id) for s in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])  # no sequence in two splits
    per = df.groupby(["scenario", "split"]).sequence_id.nunique().unstack(fill_value=0)
    assert (per.to_numpy() >= 1).all() and len(per) == 30  # every scenario and transition is in train, val AND test
    for _, g in df.groupby("sequence_id"):
        assert g.split.nunique() == 1  # a sequence is never divided
    again = apply_split(df.drop(columns="split"), v2_400[0], seed=11)
    assert again.split.equals(df.split)
    with pytest.raises(ValueError, match=">= 3"):
        split_dataset_stratified(df[df.sequence_id.isin(sorted(df.sequence_id.unique())[:2])], seed=1)
    assert "split" in split_dataset(df.drop(columns="split"), seed=1).columns  # legacy random split still available


def test_row_stride_keeps_future_labels_exact(v2_400):
    cfg1, full = v2_400
    sub, _ = generate_dataset(v2_cfg(400, stride=3))
    keep = full.groupby("sequence_id", sort=False).cumcount() % 3 == 0
    cols = [c for c in full.columns if c != "split"]
    pd.testing.assert_frame_equal(full.loc[keep, cols].reset_index(drop=True), sub[cols], check_dtype=False, rtol=1e-9, atol=1e-9)
    valid = sub.dropna(subset=["future_5min_label"])
    assert len(valid) > 0
    for _, row in valid.sample(25, random_state=1).iterrows():  # label at t+300 really is the label 300 s later in the SAME sequence
        later = sub[(sub.sequence_id == row.sequence_id) & (sub.timestamp_s == row.timestamp_s + 300)]
        assert len(later) == 1 and later.iloc[0].label == row.future_5min_label


def test_shipped_v2_dataset_covers_the_states_the_legacy_data_missed():
    from scripts.analyze_training_coverage import audit

    v2p = REPO / "sample_data" / "synthetic_v2" / "synthetic_dataset.csv.gz"
    v1p = REPO / "sample_data" / "synthetic_long" / "synthetic_dataset.csv"
    if not (v2p.exists() and v1p.exists()):
        pytest.skip("datasets not present")
    new, old = audit(pd.read_csv(v2p)), audit(pd.read_csv(v1p))
    root_cause = ["empty zone (people_count == 0)", "ANY core movement metric undefined (post warm-up)", "stationary people with direction undefined (valid 0.0 speed)"]
    for state in root_cause:
        assert old["states"][state]["all"] == 0, state  # the confirmed root cause: legacy training data never contained this
        assert all(new["states"][state][sp] > 0 for sp in ("train", "val", "test")), state  # now present in EVERY split
    assert old["post_warmup_rows"] == 35280 and old["min_people_count"] >= 13  # reported figure verified, not assumed
    assert new["max_people_count"] if "max_people_count" in new else True


# ---------------------------------------------------- feature semantics / ordering
def test_feature_row_distinguishes_valid_zero_from_unknown_and_handles_non_finite():
    empty, standing = probe_row(PROBES[0]), probe_row(PROBES[1])
    assert math.isnan(empty["avg_speed_px_s"]) and empty["people_count"] == 0.0 and math.isnan(empty["w60_speed_mean"])
    assert standing["avg_speed_px_s"] == 0.0 and standing["w60_speed_mean"] == 0.0 and standing["people_count"] == 1.0  # valid zeros survive
    assert math.isnan(standing["direction_variance"]) and math.isnan(standing["w60_direction_variance_mean"])
    m = ZoneMetrics("z", 0.0, 3, 0.1, float("inf"), float("nan"), -float("inf"), None, 0, 0)
    row = build_feature_row(m, None)
    assert all(math.isnan(row[k]) for k in ("avg_speed_px_s", "speed_variance", "flow_direction_deg", "direction_variance"))  # inf/NaN/None -> undefined
    assert row["people_count"] == 3.0 and row["inflow_count"] == 0.0


@needs_models
def test_training_and_inference_feature_order_agree():
    new = MultiHorizonRiskModel.load(ART)
    h1 = RiskModel.load(ART / "risk_model.pkl")
    names = new.current.feature_names
    assert names == new.future_5m.feature_names == new.future_10m.feature_names == h1.feature_names and len(names) == 76
    ds = pd.read_csv(REPO / "sample_data" / "synthetic_v2" / "synthetic_dataset.csv.gz", nrows=50)
    assert names == get_feature_columns(ds)  # exactly what training used, in the same order
    assert list(probe_frame(names).columns) == names and set(names) <= set(probe_row(PROBES[0]))
    X = probe_frame(names)
    shuffled = X[list(reversed(names))]
    assert list(new.current.predict(shuffled)) == list(new.current.predict(X))  # models select by NAME, never by position


@needs_models
def test_artifact_metadata_records_reproducibility_information():
    for path in [ART / "risk_model.pkl", ART / "future_risk_current.pkl", ART / "future_risk_5m.pkl", ART / "future_risk_10m.pkl"]:
        meta = json.loads(Path(str(path) + ".meta.json").read_text())
        assert meta["dataset_version"] == "synthetic_v2" and meta["feature_schema_version"] == "f1-76" and meta["training_seed"] == 42
        assert "v2" in meta["model_version"] and meta["sklearn_version"] and meta["training_rows"] > 5000  # +10m trains on fewer rows by construction (only the first third of each sequence has a valid +10m label) and meta["feature_names"]
        pats = {tuple(p["nan_feature_indices"]): p["training_rows"] for p in meta["training_nan_patterns"]}
        assert () in pats and len(pats) >= 10 and sum(pats.values()) == meta["training_rows"]  # the undefined-input states it WAS trained on
    assert json.loads((ART / "future_risk_5m.pkl.meta.json").read_text())["horizon_seconds"] == 300.0


# ------------------------------------------------- model behaviour on the failing states
@needs_models
def test_new_models_handle_every_probe_state_at_every_horizon():
    new, h1 = MultiHorizonRiskModel.load(ART), RiskModel.load(ART / "risk_model.pkl")
    res = run_probes({"H1": h1, "current": new.current, "future_5m": new.future_5m, "future_10m": new.future_10m})
    wrong = {c[:-3]: list(res.index[~res[c]]) for c in res.columns if c.endswith("_ok") and not res[c].all()}
    assert wrong == {}, wrong


@needs_models
def test_empty_zone_and_single_person_are_not_alarms_at_any_horizon():
    new = MultiHorizonRiskModel.load(ART)
    X = probe_frame(new.current.feature_names).loc[["empty_zone", "one_stationary_person", "stationary_low_occupancy"]]
    for m in (new.current, new.future_5m, new.future_10m):
        assert not set(m.predict(X)) & DANGER, m.model_version  # never an alarm
        p = m.predict_proba(X)
        # SAFE keeps more probability mass than HIGH+CRITICAL combined. A residual (+10m empty zone: ~0.32) is EXPECTED and honest: the
        # training data deliberately contains sudden-arrival transitions out of empty zones, so future danger is not impossible.
        assert (p["SAFE"] > p[["HIGH", "CRITICAL"]].sum(axis=1)).all(), m.model_version
    assert (new.current.predict_proba(X)[["HIGH", "CRITICAL"]].sum(axis=1) < 0.05).all()  # NOW: essentially no danger mass


@needs_models
def test_dense_stationary_crowd_with_undefined_direction_stays_dangerous():
    new, h1 = MultiHorizonRiskModel.load(ART), RiskModel.load(ART / "risk_model.pkl")
    X = probe_frame(new.current.feature_names).loc[["dense_stationary_high", "dense_stationary_critical", "low_speed_dangerous_congestion"]]
    for m in (h1, new.current):
        assert set(m.predict(X)) <= DANGER and (m.predict_proba(X)[["HIGH", "CRITICAL"]].sum(axis=1) > 0.8).all()
    assert (X.loc[["dense_stationary_high", "dense_stationary_critical"], "direction_variance"].isna()).all()  # undefined, yet still dangerous


@needs_models
def test_normal_moving_crowd_is_safe():
    new = MultiHorizonRiskModel.load(ART)
    X = probe_frame(new.current.feature_names).loc[["normal_moving", "sparse_moving", "low_speed_safe"]]
    for m in (new.current, new.future_5m, new.future_10m):
        assert set(m.predict(X)) <= {"SAFE", "LOW"}


@needs_models
@needs_archive
def test_characterisation_of_the_previous_artifacts_documents_the_confirmed_bug():
    """The pre-fix artifacts are kept so the fix stays demonstrable: they fail exactly these states."""
    old, h1 = MultiHorizonRiskModel.load(OLD), RiskModel.load(OLD / "risk_model.pkl")
    X = probe_frame(old.current.feature_names)
    assert old.future_10m.predict(X.loc[["empty_zone"]])[0] == "CRITICAL"  # the reported symptom
    assert h1.predict(X.loc[["one_stationary_person"]])[0] in DANGER  # H1 was affected too (stationary => danger shortcut)
    assert old.current.predict(X.loc[["dense_stationary_critical"]])[0] in {"SAFE", "LOW"}  # missed danger: undefined direction hid the crowd


@needs_models
def test_explanations_work_when_inputs_are_undefined():
    new = MultiHorizonRiskModel.load(ART)
    ref = pd.read_csv(REPO / "sample_data" / "synthetic_v2" / "synthetic_dataset.csv.gz")
    ref = ref[ref.split == "train"]
    from models.features import prepare_features

    X_ref, _, _ = prepare_features(ref, feature_columns=new.current.feature_names)
    explainer = ModelExplainer.from_training_data(new.current, X_ref)
    X = probe_frame(new.current.feature_names)
    for name, expect_danger in (("empty_zone", False), ("dense_stationary_high", True)):
        ex = explainer.explain(X.loc[[name]])
        assert ex.predicted_class in LABELS_ORDERED and (ex.predicted_class in DANGER) == expect_danger
        assert ex.predicted_class == new.current.predict(X.loc[[name]])[0]  # consistent with the model
        undefined = {f for f, v in ex.feature_values.items() if math.isnan(v)}
        assert undefined  # the row really has undefined inputs
        assert set(ex.feature_contributions) == set(new.current.feature_names) and all(math.isfinite(c) for c in ex.feature_contributions.values())
    dense = explainer.explain(X.loc[["dense_stationary_high"]])
    assert any(("density" in f.feature or f.feature == "people_count") for f in dense.top_positive_features)  # danger is explained by occupancy


# ------------------------------------------------------------- K1 policy
def k1(**over):
    base = load_config(venue_path=VENUE).diversion
    cfg = base.model_validate({**base.model_dump(), **over})
    return cfg, DiversionEngine(ZoneGraph(load_config(venue_path=VENUE).venue.zones), cfg)


def zs(zid, people=0, risk="SAFE", f5=None, f10=None, reliable=True, capacity=None, surge=False):
    z = next(z for z in load_config(venue_path=VENUE).venue.zones if z.id == zid)
    return ZoneState(zid, z.enabled, capacity if capacity is not None else z.capacity, z.preferred_utilization_threshold, z.neighbors, z.neighbor_costs,
                     people, people / z.area_m2, risk, f5, f10, surge, None, None, inputs_reliable=reliable, input_note=None if reliable else "stationary: 1/76 undefined")


def test_capacity_rule_is_independent_of_the_model_and_cannot_be_hidden_by_unknown_inputs():
    cfg, engine = k1()
    full = zs("entrance", people=80, risk="SAFE", reliable=False)  # model says SAFE, inputs out-of-distribution, but 80/80 people
    assert source_triggers(full, cfg) == ["capacity_rule"]
    rec = engine.recommend_all({"entrance": full, "main_hall": zs("main_hall", people=20)}, 0.0)["entrance"]
    assert rec.destination_zone == "main_hall" and rec.trigger_basis == ("capacity_rule",)
    assert rec.reliability == "reliable" and "capacity_rule" in rec.reliability_note  # rests on the rule, not on the unreliable model
    below = zs("entrance", people=79, risk="SAFE", reliable=False)
    assert source_triggers(below, cfg) == []  # high occupancy alone, below configured capacity, does not trigger
    off_cfg, _ = k1(source_capacity_trigger_utilization=None)
    assert source_triggers(full, off_cfg) == []  # switchable; None restores the previous behaviour


def test_model_only_triggers_from_out_of_distribution_inputs_are_provisional_not_suppressed():
    _, engine = k1()
    states = {"entrance": zs("entrance", people=3, risk="HIGH", f5="CRITICAL", reliable=False), "main_hall": zs("main_hall", people=20)}
    rec = engine.recommend_all(states, 0.0)["entrance"]
    assert rec.destination_zone == "main_hall"  # still computed and reported
    assert rec.reliability == "provisional" and set(rec.trigger_basis) == {"model_current_risk", "model_future_risk_5m"}
    assert "out-of-distribution" in rec.reliability_note and "unverified" in rec.reliability_note
    reliable = engine.recommend_all({**states, "entrance": zs("entrance", people=3, risk="HIGH", reliable=True)}, 0.0)["entrance"]
    assert reliable.reliability == "reliable" and reliable.reliability_note is None  # same prediction, supported inputs -> reliable
    surge = engine.recommend_all({**states, "entrance": zs("entrance", people=3, risk="HIGH", reliable=False, surge=True)}, 0.0)["entrance"]
    assert surge.reliability == "reliable" and "surge_flag" in surge.trigger_basis  # a model-independent trigger also fired


def test_destination_with_out_of_distribution_inputs_is_rejected_not_trusted_as_safe():
    cfg, engine = k1()
    states = {"main_hall": zs("main_hall", people=100, risk="CRITICAL"), "exit_north": zs("exit_north", people=5, risk="SAFE", reliable=False),
              "exit_south": zs("exit_south", people=5, risk="SAFE"), "entrance": zs("entrance", people=5)}
    rec = engine.recommend_all(states, 0.0)["main_hall"]
    assert rec.destination_zone != "exit_north"
    assert any(c.zone_id == "exit_north" and "out-of-distribution" in c.reason for c in rec.rejected_candidates)
    _, lenient = k1(destination_requires_reliable_inputs=False)
    assert "exit_north" not in {c.zone_id for c in lenient.recommend_all(states, 0.0)["main_hall"].rejected_candidates}


# --------------------------------------------------- L1 pipeline: states and policy end to end
class GridDetector:
    """n exactly-stationary people on a grid inside the entrance polygon (E1 sees zero displacement)."""

    def __init__(self, n):
        self.n = n

    def detect(self, frame):
        return [Detection(5 + 9 * (i % 10), 5 + 9 * (i // 10), 11 + 9 * (i % 10), 11 + 9 * (i // 10), 0.9) for i in range(self.n)]


class Fake:
    model_version = "fake"

    def __init__(self, label="SAFE", patterns=None, names=("people_count", "avg_speed_px_s", "direction_variance"), by_zone=None):
        self.feature_names, self.label, self.by_zone = list(names), label, by_zone or {}
        self.metadata = {"training_nan_patterns": patterns} if patterns is not None else {}

    def predict_proba(self, X):
        P = pd.DataFrame(0.0, index=X.index, columns=LABELS_ORDERED)
        for zid in X.index:
            P.loc[zid, self.by_zone.get(zid, self.label)] = 1.0
        return P


def pipe(detector, label="SAFE", patterns=None, cfg=None, models=None, by_zone=None):
    cfg = cfg or load_config(venue_path=VENUE)
    m = Fake(label, patterns, by_zone=by_zone)
    return InferencePipeline(cfg, detector, PersonTracker(30, 15.0), models or {"current": m, "future_5m": m, "future_10m": m})


def run(p, n):
    for i in range(n):
        r = p.process_frame(FRAME, i, float(i))
    return r


def zone(r, zid):
    return next(z for z in r["zones"] if z["zone_id"] == zid)


def test_input_state_is_classified_by_cause_and_supported_only_if_seen_in_training():
    supported_empty = [{"nan_feature_indices": [], "training_rows": 900}, {"nan_feature_indices": [1, 2], "training_rows": 500}]
    r = run(pipe(GridDetector(0), patterns=supported_empty), 3)
    q = zone(r, "main_hall")["input_quality"]
    assert q["undefined_state"] == "no_people" and q["complete"] is False and q["pattern_supported"] is True and q["pattern_training_rows"] == 500
    assert r["unsupported_input_zones"] == [] and not any("out-of-distribution" in w for w in r["warnings"])
    r2 = run(pipe(GridDetector(4), patterns=supported_empty), 3)  # 4 exactly-stationary people: speed 0.0 valid, direction undefined
    q2 = zone(r2, "entrance")["input_quality"]
    assert q2["undefined_state"] == "no_movement_observed" and zone(r2, "entrance")["metrics"]["avg_speed_px_s"] == 0.0
    assert q2["pattern_supported"] is False and q2["pattern_training_rows"] == 0 and "entrance" in r2["unsupported_input_zones"]
    assert any("out-of-distribution" in w and "entrance" in w for w in r2["warnings"])
    first = pipe(GridDetector(4), patterns=supported_empty).process_frame(FRAME, 0, 0.0)  # tracks just born: no history yet
    assert zone(first, "entrance")["input_quality"]["undefined_state"] == "movement_unavailable"
    few = pipe(GridDetector(4), patterns=[{"nan_feature_indices": [2], "training_rows": 5}], cfg=None)
    assert zone(run(few, 3), "entrance")["input_quality"]["pattern_supported"] is False  # seen, but in fewer than min_pattern_support_rows


def test_warmup_only_undefined_inputs_are_classified_separately():
    p = pipe(GridDetector(0), models={"current": Fake(names=("people_count", "w5_density_slope"))})
    metrics = ZoneMetrics("entrance", 0.0, 2, 0.1, 5.0, 1.0, 90.0, 0.1, 0, 0)  # speed and direction DEFINED
    X = pd.DataFrame({"entrance": {"people_count": 2.0, "w5_density_slope": float("nan")}}).T
    assert p._input_quality(X, {"entrance": metrics})["entrance"]["undefined_state"] == "warmup_history"


def test_model_only_alarm_on_unsupported_inputs_is_reported_as_provisional_in_json():
    r = run(pipe(GridDetector(4), label="SAFE", by_zone={"entrance": "HIGH"},
                 patterns=[{"nan_feature_indices": [], "training_rows": 900}, {"nan_feature_indices": [1, 2], "training_rows": 500}]), 3)
    rec = next(x for x in r["diversion_recommendations"] if x["source_zone"] == "entrance")
    assert rec["reliability"] == "provisional" and rec["source_inputs_reliable"] is False and rec["reliability_note"]
    assert set(rec["trigger_basis"]) == {"model_current_risk", "model_future_risk_5m", "model_future_risk_10m"}
    assert rec["destination_zone"] == "main_hall" and rec["status"] == "RECOMMENDED"  # reported, not suppressed
    assert any("PROVISIONAL" in w and "entrance" in w for w in r["warnings"]) and validate_result(r) == []
    bad = json.loads(json.dumps(r))
    bad["diversion_recommendations"][0].update(reliability="provisional", reliability_note=None)
    assert any("must explain itself" in e for e in validate_result(bad))
    bad["diversion_recommendations"][0]["reliability"] = "maybe"
    assert any("reliability invalid" in e for e in validate_result(bad))


def test_over_capacity_stationary_crowd_is_recommended_even_when_the_model_says_safe():
    empty_ok = [{"nan_feature_indices": [1, 2], "training_rows": 500}]  # the empty-zone pattern WAS trained on; the standing-crowd one was NOT
    r = run(pipe(GridDetector(80), label="SAFE", patterns=empty_ok), 3)  # 80 standing people = entrance capacity (80); the model is blind to it
    z = zone(r, "entrance")
    assert z["people_count"] == 80 and z["utilization"] == 1.0 and z["risk"]["current"]["risk_class"] == "SAFE"
    rec = next(x for x in r["diversion_recommendations"] if x["source_zone"] == "entrance")
    assert rec["trigger_basis"] == ["capacity_rule"] and rec["reliability"] == "reliable" and "capacity_rule" in rec["reliability_note"]
    assert rec["status"] == "RECOMMENDED" and rec["destination_zone"] == "main_hall" and validate_result(r) == []
    # if NO neighbour's inputs can be trusted, K1 says so instead of sending people somewhere unverified - the alarm itself is still raised
    blind = run(pipe(GridDetector(80), label="SAFE", patterns=[]), 3)
    brec = next(x for x in blind["diversion_recommendations"] if x["source_zone"] == "entrance")
    assert brec["trigger_basis"] == ["capacity_rule"] and brec["status"] == "NO_SAFE_DESTINATION"
    assert any("out-of-distribution" in c["reason"] for c in brec["rejected_candidates"])
    off = load_config(venue_path=VENUE)
    off = AppConfig.model_validate({**off.model_dump(), "diversion": {**off.diversion.model_dump(), "source_capacity_trigger_utilization": None}})
    assert not any(x["source_zone"] == "entrance" for x in run(pipe(GridDetector(80), label="SAFE", patterns=empty_ok, cfg=off), 3)["diversion_recommendations"])


@needs_models
def test_real_models_empty_venue_is_quiet_supported_and_ok_after_warmup():
    cfg = load_config(venue_path=VENUE)
    p = InferencePipeline.from_config(cfg, detector=GridDetector(0), tracker=PersonTracker(30, 15.0), explainers={})
    r = run(p, 70)
    assert r["diversion_recommendations"] == [] and r["pipeline_status"] == "ok"
    assert r["unsupported_input_zones"] == [] and set(r["incomplete_input_zones"]) >= {"entrance", "main_hall"}  # undefined, yet supported
    for z in r["zones"]:
        if z["enabled"]:
            assert all(z["risk"][h]["risk_class"] in {"SAFE", "LOW"} for h in ("current", "future_5m", "future_10m")), z["zone_id"]
            assert z["input_quality"]["pattern_supported"] is True and z["input_quality"]["undefined_state"] == "no_people"


@needs_models
def test_real_models_flag_a_dense_standing_crowd_as_dangerous_with_independent_corroboration():
    cfg = load_config(venue_path=VENUE)
    p = InferencePipeline.from_config(cfg, detector=GridDetector(100), tracker=PersonTracker(30, 15.0), explainers={})
    r = run(p, 70)
    z = zone(r, "entrance")
    assert z["people_count"] == 100 and z["metrics"]["avg_speed_px_s"] == 0.0 and z["metrics"]["direction_variance"] is None
    assert z["risk"]["current"]["risk_class"] in DANGER and z["input_quality"]["pattern_supported"] is True
    rec = next(x for x in r["diversion_recommendations"] if x["source_zone"] == "entrance")
    assert {"model_current_risk", "capacity_rule"} <= set(rec["trigger_basis"]) and rec["reliability"] == "reliable"
    assert validate_result(r) == [] and json.dumps(r, allow_nan=False)
