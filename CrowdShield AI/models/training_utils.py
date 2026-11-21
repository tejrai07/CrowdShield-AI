"""Shared training utilities for H1 (current-risk) and I1 (future-risk)
model training scripts.

Extracted from scripts/train_risk_model.py during I1 so the two training
scripts share one implementation of leakage auditing, pipeline
construction, latency benchmarking, and safety-first model selection —
per the "do not duplicate utilities" requirement. Behavior is unchanged
from H1's original inline versions (verified by re-running H1's full
training + test suite after the extraction).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.features import EXCLUDED_COLUMNS
from models.risk_model import RiskModel

SELECTION_METRICS = ("dangerous_recall", "macro_f1", "false_safe_rate", "latency_ms", "model_size_kb", "accuracy")


def leakage_audit(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """Mandatory H1/I1 check: excluded metadata/target columns must never
    appear in the feature list, and train/val/test sequence IDs must be
    fully disjoint (re-verifying G1's own guarantee, not trusting it blindly)."""
    leaked = EXCLUDED_COLUMNS & set(feature_columns)
    if leaked:
        raise AssertionError(f"LEAKAGE: excluded columns found in feature set: {leaked}")

    seq_by_split = {
        split: set(df[df["split"] == split]["sequence_id"]) for split in ["train", "val", "test"]
    }
    for a in ["train", "val", "test"]:
        for b in ["train", "val", "test"]:
            if a >= b:
                continue
            overlap = seq_by_split[a] & seq_by_split[b]
            if overlap:
                raise AssertionError(f"LEAKAGE: {a} and {b} share sequence_ids: {overlap}")

    print("Leakage audit: PASS (no excluded columns in features; train/val/test sequence IDs disjoint)")


def build_lr_pipeline(C: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=C, random_state=42)),
    ])


def build_hgb_pipeline(max_leaf_nodes: int) -> Pipeline:
    # HistGradientBoostingClassifier handles NaN natively (F1's warm-up
    # convention) — no imputer step needed, deliberately kept separate
    # from the LR pipeline's preprocessing per the spec's guidance.
    return Pipeline([
        ("clf", HistGradientBoostingClassifier(max_leaf_nodes=max_leaf_nodes, random_state=42)),
    ])


def measure_inference_latency(model: RiskModel, X: pd.DataFrame, n_single: int = 200) -> dict:
    single_row = X.iloc[[0]]
    # warm-up
    for _ in range(5):
        model.predict_risk_score(single_row)

    t0 = time.perf_counter()
    for _ in range(n_single):
        model.predict_risk_score(single_row)
    single_ms = (time.perf_counter() - t0) / n_single * 1000

    t0 = time.perf_counter()
    model.predict_risk_score(X)
    batch_elapsed = time.perf_counter() - t0
    batch_ms_per_row = batch_elapsed / len(X) * 1000

    return {
        "single_sample_latency_ms": round(single_ms, 4),
        "batch_latency_ms_per_row": round(batch_ms_per_row, 5),
        "batch_total_ms": round(batch_elapsed * 1000, 2),
        "batch_n_rows": len(X),
    }


def model_size_kb(path: Path) -> float:
    return round(path.stat().st_size / 1024, 1)


def cleanup_tmp_artifact(path: Path) -> None:
    """Remove a temporary candidate model's pipeline file AND its
    metadata sidecar (RiskModel.save() always writes both — see
    models/risk_model.py's portability fix)."""
    path.unlink(missing_ok=True)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.unlink(missing_ok=True)


def select_best(candidates: list[dict]) -> dict:
    """Safety-first selection: dangerous_recall desc, macro_f1 desc,
    false_safe_rate asc, latency asc, model_size asc, accuracy desc.

    Each candidate dict must have "val_metrics" (with dangerous_recall,
    macro_f1, false_safe_rate, accuracy), "latency" (with
    single_sample_latency_ms), and "size_kb".
    """
    def sort_key(c):
        return (
            -c["val_metrics"]["dangerous_recall"],
            -c["val_metrics"]["macro_f1"],
            c["val_metrics"]["false_safe_rate"],
            c["latency"]["single_sample_latency_ms"],
            c["size_kb"],
            -c["val_metrics"]["accuracy"],
        )
    return sorted(candidates, key=sort_key)[0]


FEATURE_SCHEMA_VERSION = "f1-76"  # the 76 model inputs defined by models/features.py (E1 metrics + F1 windows); unchanged by the reliability fix


def nan_pattern_support(X: pd.DataFrame) -> list[dict]:
    """Which EXACT sets of undefined (NaN) features occurred in the training matrix, and in how many rows.

    Indices refer to X's column order (== the model's feature_names order). Stored in the model metadata so
    inference can tell 'undefined but the model was trained on this state' from 'undefined in a way the model
    never saw' (out-of-distribution) - see inference/pipeline.py input_quality."""
    patterns, counts = np.unique(X.isna().to_numpy(), axis=0, return_counts=True)
    out = [{"nan_feature_indices": [int(i) for i in np.flatnonzero(row)], "training_rows": int(n)} for row, n in zip(patterns, counts)]
    return sorted(out, key=lambda d: -d["training_rows"])
