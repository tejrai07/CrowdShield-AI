"""Old-vs-new risk-model evaluation for the post-L1 reliability fix. Same metrics, same frozen splits for both generations.

  python scripts/evaluate_reliability.py [--output models/artifacts/reliability_report.json]

Metric definitions (all computed here, from labels - nothing assumed):
  dangerous_recall   = |true in {HIGH,CRITICAL} and pred in {HIGH,CRITICAL}| / |true in {HIGH,CRITICAL}|
  false_safe_rate    = |true in {HIGH,CRITICAL} and pred in {SAFE,LOW}|      / |true in {HIGH,CRITICAL}|   (missed danger)
  false_alarm_rate   = |true in {SAFE,LOW}      and pred in {HIGH,CRITICAL}| / |true in {SAFE,LOW}|        (needless alarm)
  escalation_recall  = of rows NOT dangerous now that truly ARE dangerous at the horizon, the share predicted dangerous at that horizon
Evaluation sets: V2-TEST (the held-out test sequences of data/scenarios_v2.yaml - never seen by the new models; the old models were
trained on different data) and LEGACY-TEST (original I1/H1 test sequences; verified to share no feature vector with v2 train).
Synthetic data only: near-perfect numbers measure how cleanly the generator separates classes, NOT real-world skill.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evaluation.metrics import compute_all_metrics, escalation_recall  # noqa: E402
from evaluation.probes import run_probes  # noqa: E402
from models.features import DANGEROUS_LABELS, LABELS_ORDERED, get_feature_columns  # noqa: E402
from models.future_risk import MultiHorizonRiskModel, prepare_future_features  # noqa: E402
from models.risk_model import RiskModel  # noqa: E402
from models.training_utils import measure_inference_latency  # noqa: E402

ART = REPO_ROOT / "models" / "artifacts"
OLD = ART / "archive_pre_reliability"
DANGER, NOT_DANGER = list(DANGEROUS_LABELS), ["SAFE", "LOW"]
SCENARIO_FOCUS = ["EMPTY_ZONE", "SINGLE_STATIONARY_PERSON", "STATIONARY_LOW_OCCUPANCY", "STATIONARY_MODERATE_DENSITY",
                  "STATIONARY_DENSE_HIGH", "STATIONARY_DENSE_CRITICAL", "MOVING_DENSE_HIGH", "SLOW_WALK_SAFE", "DROPOUT_DENSE_HIGH", "NORMAL_STABLE"]


def rates(y: pd.Series, pred: pd.Series) -> dict:
    d, s = y.isin(DANGER), y.isin(NOT_DANGER)
    return {"rows": int(len(y)), "accuracy": float((y == pred).mean()),
            "dangerous_rows": int(d.sum()), "dangerous_recall": float(pred[d].isin(DANGER).mean()) if d.any() else None,
            "false_safe_rate": float(pred[d].isin(NOT_DANGER).mean()) if d.any() else None,
            "safe_or_low_rows": int(s.sum()), "false_alarm_rate": float(pred[s].isin(DANGER).mean()) if s.any() else None}


def eval_one(model, df: pd.DataFrame, horizon: str, with_cm: bool = False) -> dict:
    X, y = prepare_future_features(df, horizon, get_feature_columns(df))
    proba = model.predict_proba(X)
    pred = proba.idxmax(axis=1)
    m = compute_all_metrics(y, pred, proba.values)
    out = {k: m[k] for k in ("accuracy", "macro_f1", "weighted_f1", "dangerous_recall", "false_safe_rate", "ordinal_mae", "log_loss") if k in m}
    out.update({"rows": int(len(y)), "false_alarm_rate": rates(y, pred)["false_alarm_rate"]})
    if horizon != "current":
        out["escalation_recall"] = escalation_recall(df.loc[X.index, "label"], y, pred)
    if with_cm:
        out["confusion_matrix"] = {"labels": list(LABELS_ORDERED), "matrix": m.get("confusion_matrix"), "rows": "true class", "columns": "predicted class"}
        out["per_class"] = m.get("per_class")
    out["_X_y_pred"] = (X, y, pred)
    return out


def strip(d):
    return {k: v for k, v in d.items() if k != "_X_y_pred"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="models/artifacts/reliability_report.json")
    args = ap.parse_args()

    v2 = pd.read_csv(REPO_ROOT / "sample_data/synthetic_v2/synthetic_dataset.csv.gz")
    v1 = pd.read_csv(REPO_ROOT / "sample_data/synthetic_long/synthetic_dataset.csv")
    h1d = pd.read_csv(REPO_ROOT / "sample_data/synthetic/synthetic_dataset.csv")
    sets = {"v2_test": v2[v2.split == "test"], "legacy_i1_test": v1[v1.split == "test"], "legacy_h1_test": h1d[h1d.split == "test"]}

    t0 = time.perf_counter(); old = MultiHorizonRiskModel.load(OLD); old_h1 = RiskModel.load(OLD / "risk_model.pkl"); old_load = time.perf_counter() - t0
    t0 = time.perf_counter(); new = MultiHorizonRiskModel.load(ART); new_h1 = RiskModel.load(ART / "risk_model.pkl"); new_load = time.perf_counter() - t0
    gens = {"old": {"H1": old_h1, "I1_current": old.current, "I1_5m": old.future_5m, "I1_10m": old.future_10m},
            "new": {"H1": new_h1, "I1_current": new.current, "I1_5m": new.future_5m, "I1_10m": new.future_10m}}
    hz = {"H1": "current", "I1_current": "current", "I1_5m": "5m", "I1_10m": "10m"}

    report: dict = {"definitions": __doc__, "model_versions": {g: {k: m.model_version for k, m in ms.items()} for g, ms in gens.items()}, "overall": {}, "per_scenario_v2_test": {}, "complete_vs_incomplete_v2_test": {}}
    cache: dict = {}
    for set_name, df in sets.items():
        report["overall"][set_name] = {}
        for gen, ms in gens.items():
            for name, model in ms.items():
                if set_name == "legacy_h1_test" and name != "H1":
                    continue
                if set_name == "legacy_i1_test" and name == "H1":
                    continue
                r = eval_one(model, df, hz[name], with_cm=(set_name == "v2_test"))
                cache[(set_name, gen, name)] = r["_X_y_pred"]
                report["overall"][set_name].setdefault(name, {})[gen] = strip(r)

    # per scenario (v2 test): where do the generations differ?
    for name in ("H1", "I1_current", "I1_5m", "I1_10m"):
        report["per_scenario_v2_test"][name] = {}
        for gen in ("old", "new"):
            X, y, pred = cache[("v2_test", gen, name)]
            scen = v2.loc[X.index, "scenario"]
            for sc in sorted(scen.unique()):
                mask = scen == sc
                report["per_scenario_v2_test"][name].setdefault(sc, {})[gen] = rates(y[mask], pred[mask])

    # complete vs incomplete inputs (v2 test, current-horizon models)
    for name in ("H1", "I1_current"):
        for gen in ("old", "new"):
            X, y, pred = cache[("v2_test", gen, name)]
            inc = X.isna().any(axis=1)
            report["complete_vs_incomplete_v2_test"].setdefault(name, {})[gen] = {"complete_inputs": rates(y[~inc], pred[~inc]), "incomplete_inputs": rates(y[inc], pred[inc])}
    X = cache[("v2_test", "new", "I1_current")][0]
    names = new.current.feature_names
    support = {tuple(p["nan_feature_indices"]): p["training_rows"] for p in new.current.metadata["training_nan_patterns"]}
    pats = X.isna().apply(lambda r: tuple(np.flatnonzero(r.to_numpy())), axis=1)
    rows = pats.map(lambda t: 0 if not t else support.get(t, 0))
    report["input_flags_v2_test"] = {"rows": int(len(X)), "incomplete_rows": int((pats.map(len) > 0).sum()),
                                     "unsupported_rows_under_new_models(<30 training rows)": int(((pats.map(len) > 0) & (rows < 30)).sum()),
                                     "distinct_undefined_patterns_in_test": int(pats.map(lambda t: t).nunique()), "distinct_patterns_in_training": len(support)}

    # legacy I1 test, split by the generator's ORIGINAL 120 s horizon (declared in advance, not tuned): within it the legacy labels are
    # consistent with density; beyond it the legacy generator extrapolated trends (up to 5,144 people in 50 m2) while labels stayed constant.
    report["legacy_i1_test_by_time"] = {}
    for name in ("I1_current", "I1_5m", "I1_10m"):
        for gen in ("old", "new"):
            X, y, pred = cache[("legacy_i1_test", gen, name)]
            early = v1.loc[X.index, "timestamp_s"] <= 120
            for tag, mask in (("t<=120s", early), ("t>120s", ~early)):
                report["legacy_i1_test_by_time"].setdefault(name, {}).setdefault(tag, {})[gen] = rates(y[mask], pred[mask])
    lt = v1[(v1.split == "test") & (v1.timestamp_s > 120)]
    low = lt[lt.label.isin(["SAFE", "LOW"])]
    report["legacy_label_density_inconsistency"] = {
        "legacy_test_rows_t>120s": int(len(lt)), "SAFE_or_LOW_labelled_rows": int(len(low)),
        "of_which_density>=1.5_people_per_m2 (G1's own HIGH regime)": int((low.density >= 1.5).sum()),
        "max_people_count_among_SAFE_or_LOW": int(low.people_count.max()) if len(low) else None}

    # probes + latency/size
    report["probes"] = {}
    pr = run_probes({f"{g}_{k}": m for g, ms in gens.items() for k, m in ms.items()})
    report["probes"] = {"within_expected_class_set_of_13": {c[:-3]: int(pr[c].sum()) for c in pr.columns if c.endswith("_ok")}, "predictions": pr[[c for c in pr.columns if not c.endswith("_ok")]].to_dict(orient="index")}
    Xp = cache[("v2_test", "new", "I1_current")][0].head(2000)
    report["latency_and_size"] = {"bundle_load_s": {"old": round(old_load, 3), "new": round(new_load, 3)},
        "single_sample_ms": {g: {k: measure_inference_latency(m, Xp)["single_sample_latency_ms"] for k, m in ms.items()} for g, ms in gens.items()},
        "size_kb": {g: {k: round((ART if g == "new" else OLD).joinpath({"H1": "risk_model.pkl", "I1_current": "future_risk_current.pkl", "I1_5m": "future_risk_5m.pkl", "I1_10m": "future_risk_10m.pkl"}[k]).stat().st_size / 1024, 1) for k in ms} for g, ms in gens.items()}}

    out = REPO_ROOT / args.output
    out.write_text(json.dumps(report, indent=2, default=lambda o: None if o != o else float(o)))

    f = lambda v: "  n/a " if v is None else f"{v:6.3f}"
    for set_name, per in report["overall"].items():
        print(f"\n=== {set_name} ===  (rows: {next(iter(next(iter(per.values())).values()))['rows']}+)")
        print(f"{'model':<11}{'gen':<5}{'acc':>7}{'macroF1':>8}{'dangRec':>8}{'falseSafe':>10}{'falseAlarm':>11}{'escRec':>8}{'logloss':>8}")
        for name, gs in per.items():
            for gen, m in gs.items():
                print(f"{name:<11}{gen:<5}{f(m['accuracy'])}{f(m['macro_f1'])}{f(m['dangerous_recall'])}  {f(m['false_safe_rate'])}   {f(m['false_alarm_rate'])}   {f(m.get('escalation_recall'))}{f(m['log_loss'])}")
    print("\n=== per scenario (v2 test) - I1_current: false_alarm_rate / false_safe_rate / accuracy, old -> new ===")
    for sc in SCENARIO_FOCUS:
        o, n = report["per_scenario_v2_test"]["I1_current"][sc]["old"], report["per_scenario_v2_test"]["I1_current"][sc]["new"]
        g = lambda d, k: "n/a" if d[k] is None else f"{d[k]:.2f}"
        print(f"{sc:<30} alarm {g(o,'false_alarm_rate')}->{g(n,'false_alarm_rate')} | missed {g(o,'false_safe_rate')}->{g(n,'false_safe_rate')} | acc {o['accuracy']:.2f}->{n['accuracy']:.2f}  (rows {n['rows']})")
    print("\n=== legacy I1 test split by time (old -> new): false_alarm / false_safe / accuracy ===")
    for name in ("I1_current", "I1_5m", "I1_10m"):
        for tag in ("t<=120s", "t>120s"):
            o, n = report["legacy_i1_test_by_time"][name][tag]["old"], report["legacy_i1_test_by_time"][name][tag]["new"]
            g = lambda d, k: "n/a" if d[k] is None else f"{d[k]:.3f}"
            print(f"{name:<11}{tag:<8} alarm {g(o,'false_alarm_rate')}->{g(n,'false_alarm_rate')} | missed {g(o,'false_safe_rate')}->{g(n,'false_safe_rate')} | acc {o['accuracy']:.3f}->{n['accuracy']:.3f} (rows {n['rows']})")
    print("legacy label/density inconsistency:", report["legacy_label_density_inconsistency"])
    print("\ncomplete vs incomplete inputs (v2 test, I1_current):", {gen: {k: (round(v['accuracy'], 3), v['rows']) for k, v in d.items()} for gen, d in report["complete_vs_incomplete_v2_test"]["I1_current"].items()})
    print("input flags (v2 test):", report["input_flags_v2_test"])
    print("probes in expected set (/13):", report["probes"]["within_expected_class_set_of_13"])
    print("latency/size:", json.dumps(report["latency_and_size"]))


if __name__ == "__main__":
    main()
