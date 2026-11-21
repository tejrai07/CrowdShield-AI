"""I1 CLI: train, evaluate, and save the current/+5m/+10m risk models.

Usage:
    python scripts/train_future_risk.py \
        --dataset sample_data/synthetic_long/synthetic_dataset.csv \
        --output-dir models/artifacts

IMPORTANT: this trains a NEW "current" baseline (on the longer I1 dataset,
for apples-to-apples comparison with the +5m/+10m models trained on the
same dataset) — it does NOT touch H1's own `models/artifacts/risk_model.pkl`,
which remains the canonical current-risk model trained on G1's original
120s dataset. I1's current-horizon model is saved separately as
`future_risk_current.pkl` specifically for this comparison.

Uses ONLY HistGradientBoostingClassifier (per the I1 spec: "Start with the
H1 model family... do not blindly assume the H1 hyperparameters are
optimal — evaluate a small candidate set only"), with a minimal
max_leaf_nodes sweep, reusing H1's leakage audit / latency benchmark /
safety-first selection helpers from models/training_utils.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sklearn.inspection import permutation_importance  # noqa: E402

from data.synthetic_generator import load_dataset  # noqa: E402
from evaluation.metrics import escalation_recall  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from models.features import LABELS_ORDERED, get_feature_columns  # noqa: E402
from models.future_risk import HORIZON_COLUMNS, HORIZON_SECONDS, MultiHorizonRiskModel, prepare_future_features  # noqa: E402
from models.risk_model import RiskModel  # noqa: E402
from models.training_utils import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    build_hgb_pipeline,
    cleanup_tmp_artifact,
    leakage_audit,
    measure_inference_latency,
    model_size_kb,
    nan_pattern_support,
    select_best,
)

MAX_LEAF_NODES_CANDIDATES = (15, 31, 63)


def train_one_horizon(
    horizon: str, df, feature_columns: list[str], output_dir: Path, dataset_path: Path, seed: int,
    version_prefix: str = "i1", dataset_version: str = "synthetic_v1",
) -> dict:
    """Train + select + evaluate one horizon's model, mirroring H1's
    per-model workflow but restricted to the HGB family per the I1 spec."""
    X, y = prepare_future_features(df, horizon, feature_columns)
    splits = df.loc[X.index, "split"]
    train_mask, val_mask, test_mask = splits == "train", splits == "val", splits == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\n--- Horizon: {horizon} (target={HORIZON_COLUMNS[horizon]}, "
          f"+{HORIZON_SECONDS[horizon]:.0f}s) ---")
    print(f"Rows after horizon truncation: {len(X)} (train={len(X_train)} val={len(X_val)} test={len(X_test)})")

    candidates = []
    for max_leaf_nodes in MAX_LEAF_NODES_CANDIDATES:
        pipe = build_hgb_pipeline(max_leaf_nodes)
        model = RiskModel.fit(pipe, X_train, y_train, model_version=f"{version_prefix}-{horizon}")
        val_metrics = model.evaluate(X_val, y_val)
        tmp_path = output_dir / f"_tmp_{horizon}_{max_leaf_nodes}.pkl"
        model.save(tmp_path)
        candidates.append({
            "name": f"HGB(max_leaf_nodes={max_leaf_nodes})", "model": model,
            "val_metrics": val_metrics, "latency": measure_inference_latency(model, X_val),
            "size_kb": model_size_kb(tmp_path), "max_leaf_nodes": max_leaf_nodes,
        })
        cleanup_tmp_artifact(tmp_path)

    for c in candidates:
        m = c["val_metrics"]
        print(f"  {c['name']:<25} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
              f"dangerous_recall={m['dangerous_recall']:.4f} false_safe={m['false_safe_rate']:.4f}")

    selected = select_best(candidates)
    print(f"  Selected: {selected['name']}")

    test_metrics = selected["model"].evaluate(X_test, y_test)
    test_latency = measure_inference_latency(selected["model"], X_test)

    esc_recall = None
    if horizon != "current":
        y_current_true_test = df.loc[X_test.index, "label"]
        y_test_pred = selected["model"].predict(X_test)
        esc_recall = escalation_recall(y_current_true_test, y_test, y_test_pred)

    extra_metadata = {
        "horizon": horizon,
        "horizon_seconds": HORIZON_SECONDS[horizon],
        "target_column": HORIZON_COLUMNS[horizon],
        "training_dataset": str(dataset_path),
        "training_seed": seed,
        "hyperparameters": {"max_leaf_nodes": selected["max_leaf_nodes"], "random_state": 42},
        "model_family": "HistGradientBoostingClassifier",
        "dataset_version": dataset_version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "training_rows": int(len(X_train)),
        "training_nan_patterns": nan_pattern_support(X_train),
    }

    return {
        "horizon": horizon,
        "model": selected["model"],
        "extra_metadata": extra_metadata,
        "val_comparison": [
            {"name": c["name"], "val_metrics": c["val_metrics"], "latency": c["latency"], "size_kb": c["size_kb"]}
            for c in candidates
        ],
        "selected_name": selected["name"],
        "test_metrics": test_metrics,
        "test_latency": test_latency,
        "escalation_recall_test": esc_recall,
        "row_counts": {"total": len(X), "train": len(X_train), "val": len(X_val), "test": len(X_test)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CrowdShield I1 future-risk model training")
    parser.add_argument("--dataset", default="sample_data/synthetic_long/synthetic_dataset.csv")
    parser.add_argument("--output-dir", default="models/artifacts")
    parser.add_argument("--version-prefix", default="i1", help="model_version prefix, e.g. i1 or i1-v2")
    parser.add_argument("--dataset-version", default="synthetic_v1")
    args = parser.parse_args()

    logger = setup_logging("INFO")

    dataset_path = REPO_ROOT / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)
    df = load_dataset(dataset_path)
    logger.info("Loaded %d rows, %d sequences from %s", len(df), df["sequence_id"].nunique(), dataset_path)

    feature_columns = get_feature_columns(df)
    leakage_audit(df, feature_columns)

    # Sequence-boundary / temporal-leakage sanity check specific to I1:
    # confirm every non-null future label at t genuinely comes from a
    # LATER timestamp within the SAME sequence (not another sequence).
    sample = df[df["future_5min_label"].notna()].sample(min(200, df["future_5min_label"].notna().sum()), random_state=1)
    for _, row in sample.iterrows():
        target_ts = row["timestamp_s"] + 300.0
        match = df[(df["sequence_id"] == row["sequence_id"]) & (df["timestamp_s"] == target_ts)]
        assert len(match) == 1, f"future_5min_label provenance check failed for sequence {row['sequence_id']} at t={row['timestamp_s']}"
        assert match.iloc[0]["label"] == row["future_5min_label"], "future_5min_label does not match the actual label 300s later"
    print(f"Future-label provenance audit: PASS ({len(sample)} rows spot-checked against actual future rows in the same sequence)")

    output_dir = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for horizon in ("current", "5m", "10m"):
        results[horizon] = train_one_horizon(horizon, df, feature_columns, output_dir, dataset_path, seed=42, version_prefix=args.version_prefix, dataset_version=args.dataset_version)

    bundle = MultiHorizonRiskModel(
        current=results["current"]["model"], future_5m=results["5m"]["model"], future_10m=results["10m"]["model"],
    )

    # Benchmark: current-only inference vs current+5m+10m combined.
    X_bench, _ = prepare_future_features(df, "current", feature_columns)
    X_bench_row = X_bench.iloc[[0]]

    t0 = time.perf_counter()
    for _ in range(200):
        results["current"]["model"].predict_risk_score(X_bench_row)
    current_only_ms = (time.perf_counter() - t0) / 200 * 1000

    t0 = time.perf_counter()
    for _ in range(200):
        bundle.predict(X_bench_row)
    combined_ms = (time.perf_counter() - t0) / 200 * 1000

    print(f"\n=== Inference benchmark ===")
    print(f"  current-only:        {current_only_ms:.4f} ms/sample")
    print(f"  current + 5m + 10m:  {combined_ms:.4f} ms/sample")

    paths = bundle.save(output_dir, extra_metadata_by_horizon={
        "current": results["current"]["extra_metadata"],
        "future_5m": results["5m"]["extra_metadata"],
        "future_10m": results["10m"]["extra_metadata"],
    })

    report = {
        "dataset": str(dataset_path),
        "dataset_rows": len(df),
        "dataset_sequences": int(df["sequence_id"].nunique()),
        "future_label_coverage": {
            "future_5min_label": float(df["future_5min_label"].notna().mean()),
            "future_10min_label": float(df["future_10min_label"].notna().mean()),
        },
        "model_paths": {k: str(v) for k, v in paths.items()},
        "results": {
            horizon: {
                "selected_name": r["selected_name"],
                "row_counts": r["row_counts"],
                "val_comparison": r["val_comparison"],
                "test_metrics": r["test_metrics"],
                "test_latency": r["test_latency"],
                "escalation_recall_test": r["escalation_recall_test"],
                "extra_metadata": r["extra_metadata"],
            }
            for horizon, r in results.items()
        },
        "inference_benchmark_ms_per_sample": {
            "current_only": round(current_only_ms, 4),
            "current_plus_5m_plus_10m": round(combined_ms, 4),
        },
        "disclaimer": (
            "Future-risk (+5m/+10m) predictions are a MACHINE-LEARNING FORECASTING FEATURE "
            "trained on Phase G1's synthetic temporal scenarios. They are NOT physical or "
            "clinical predictions and do NOT demonstrate real-world crowd-safety forecasting "
            "accuracy. Current-risk prediction is generally more reliable than +5m, which is "
            "generally more reliable than +10m — this degradation is expected and reported "
            "honestly, not hidden."
        ),
    }
    report_path = output_dir / "future_risk_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n=== Summary (test set) ===")
    for horizon in ("current", "5m", "10m"):
        r = results[horizon]
        m = r["test_metrics"]
        esc = r["escalation_recall_test"]
        esc_str = f"{esc:.4f}" if esc is not None and esc == esc else "N/A"
        print(f"  {horizon:<8} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
              f"dangerous_recall={m['dangerous_recall']:.4f} false_safe={m['false_safe_rate']:.4f} "
              f"escalation_recall={esc_str}")

    print(f"\nModels saved to: {output_dir}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
