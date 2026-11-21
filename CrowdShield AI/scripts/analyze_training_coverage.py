"""Training-data COVERAGE audit: how often does a synthetic dataset contain each operating state the
inference pipeline can actually produce (empty zone, stationary crowd, undefined movement metrics,
abrupt changes, ...)? Every number is computed from the CSV; nothing is assumed.

    python scripts/analyze_training_coverage.py --dataset sample_data/synthetic_long/synthetic_dataset.csv [--output report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.features import DANGEROUS_LABELS, get_feature_columns  # noqa: E402

WARMUP_S = 60.0  # longest temporal window
STATIONARY_SPEED = 1.0  # px/s: below this a track is treated as not moving
LOW_SPEED = 30.0
SUDDEN_WINDOW_S = 5.0
SUDDEN_MIN_PEOPLE = 10


def predicates(d: pd.DataFrame) -> dict[str, pd.Series]:
    post = d["timestamp_s"] >= WARMUP_S
    dangerous = d["label"].isin(DANGEROUS_LABELS)
    seq = d.groupby("sequence_id", sort=False)
    spacing = float(d.groupby("sequence_id", sort=False)["timestamp_s"].diff().median())
    lag = max(1, int(round(SUDDEN_WINDOW_S / spacing)))
    before = seq["people_count"].shift(lag)
    sudden = ((d["people_count"] - before).abs() >= SUDDEN_MIN_PEOPLE) & ((d["people_count"] - before).abs() >= 0.5 * before.clip(lower=1))
    prev_label_dangerous = seq["label"].shift(1).isin(DANGEROUS_LABELS)
    prev_known = seq["label"].shift(1).notna()
    return {
        "empty zone (people_count == 0)": d["people_count"] == 0,
        "very low occupancy (people_count <= 2)": d["people_count"] <= 2,
        "stationary (people>0 and avg_speed < 1 px/s)": (d["people_count"] > 0) & (d["avg_speed_px_s"] < STATIONARY_SPEED),
        "low speed (avg_speed < 30 px/s)": d["avg_speed_px_s"] < LOW_SPEED,
        "speed metrics undefined (post warm-up)": post & d["avg_speed_px_s"].isna(),
        "direction metrics undefined (post warm-up)": post & d["direction_variance"].isna(),
        "ANY core movement metric undefined (post warm-up)": post & d[["avg_speed_px_s", "speed_variance", "direction_variance"]].isna().any(axis=1),
        "warm-up rows (t < 60 s)": ~post,
        "any NaN feature (post warm-up)": post & d[[c for c in get_feature_columns(d)]].isna().any(axis=1),
        f"sudden entry/exit (>=50% and >={SUDDEN_MIN_PEOPLE} people change within {SUDDEN_WINDOW_S:.0f} s)": sudden,
        "stationary people with direction undefined (valid 0.0 speed)": (d["people_count"] > 0) & (d["avg_speed_px_s"] == 0.0) & d["direction_variance"].isna(),
        "abrupt label jump SAFE/LOW -> HIGH/CRITICAL in one step": prev_known & ~prev_label_dangerous & dangerous & (seq["label"].shift(1).isin(["SAFE", "LOW"])),
        "dangerous label with speed < 10 px/s": dangerous & (d["avg_speed_px_s"] < 10),
        "dense & slow (density >= 1.5 and speed < 30)": (d["density"] >= 1.5) & (d["avg_speed_px_s"] < LOW_SPEED),
        "dangerous label": dangerous,
    }


def audit(df: pd.DataFrame) -> dict:
    report = {"rows": {}, "sequences": {}, "states": {}, "post_warmup_rows": int((df["timestamp_s"] >= WARMUP_S).sum()),
              "min_density": float(df["density"].min()), "min_avg_speed": float(df["avg_speed_px_s"].min()),
              "min_people_count": int(df["people_count"].min())}
    splits = ["all"] + sorted(df["split"].unique(), key=["train", "val", "test"].index)
    for sp in splits:
        d = df if sp == "all" else df[df["split"] == sp]
        report["rows"][sp], report["sequences"][sp] = int(len(d)), int(d["sequence_id"].nunique())
    for name in predicates(df):
        report["states"][name] = {}
    for sp in splits:
        d = (df if sp == "all" else df[df["split"] == sp]).reset_index(drop=True)
        for name, mask in predicates(d).items():
            report["states"][name][sp] = int(mask.sum())
    scen = df.assign(scenario=df["scenario"]).groupby(["scenario", "split"]).size().unstack(fill_value=0)
    report["rows_by_scenario_and_split"] = {k: {c: int(v) for c, v in row.items()} for k, row in scen.iterrows()}
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Audit a synthetic dataset's coverage of inference-time operating states")
    p.add_argument("--dataset", default="sample_data/synthetic_long/synthetic_dataset.csv")
    p.add_argument("--output", default=None, help="optional JSON report path")
    args = p.parse_args()
    path = Path(args.dataset) if Path(args.dataset).is_absolute() else REPO_ROOT / args.dataset
    df = pd.read_csv(path)
    rep = audit(df)
    cols = [c for c in ("train", "val", "test", "all") if c in rep["rows"]]
    print(f"dataset: {path.name} | rows {rep['rows']} | sequences {rep['sequences']} | post-warm-up rows: {rep['post_warmup_rows']}")
    print(f"min density {rep['min_density']:.3f} | min avg_speed {rep['min_avg_speed']:.2f} | min people_count {rep['min_people_count']}")
    print(f"{'operating state':<58}" + "".join(f"{c:>9}" for c in cols))
    for name, per in rep["states"].items():
        print(f"{name:<58}" + "".join(f"{per[c]:>9}" for c in cols))
    if args.output:
        out = Path(args.output) if Path(args.output).is_absolute() else REPO_ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
