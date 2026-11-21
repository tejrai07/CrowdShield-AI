"""H1 CLI: train, compare, and select the baseline risk classifier.

Usage:
    python scripts/train_risk_model.py \
        --dataset sample_data/synthetic/synthetic_dataset.csv \
        --output-dir models/artifacts

Trains Logistic Regression (with median imputation + scaling) and
HistGradientBoostingClassifier (which handles F1's warm-up NaN values
natively, no imputation needed) on the G1 dataset's fixed train split,
selects hyperparameters using the val split, evaluates BOTH candidates'
best configuration on val, selects the final model by the spec's
safety-first priority order, then evaluates the selected model exactly
once on the untouched test split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from data.synthetic_generator import load_dataset  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from models.features import (  # noqa: E402
    DANGEROUS_LABELS,
    LABELS_ORDERED,
    get_feature_columns,
    prepare_features,
)
from models.risk_model import RiskModel  # noqa: E402
from models.training_utils import (  # noqa: E402
    SELECTION_METRICS,
    FEATURE_SCHEMA_VERSION,
    build_hgb_pipeline,
    build_lr_pipeline,
    cleanup_tmp_artifact,
    leakage_audit,
    measure_inference_latency,
    model_size_kb,
    nan_pattern_support,
    select_best,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CrowdShield H1 risk model training")
    parser.add_argument("--dataset", default="sample_data/synthetic/synthetic_dataset.csv")
    parser.add_argument("--output-dir", default="models/artifacts")
    parser.add_argument("--version-tag", default="", help="inserted into model_version, e.g. -v2 -> h1-v2-hgb")
    parser.add_argument("--dataset-version", default="synthetic_v1")
    args = parser.parse_args()

    logger = setup_logging("INFO")

    dataset_path = REPO_ROOT / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)
    df = load_dataset(dataset_path)
    logger.info("Loaded %d rows, %d sequences from %s", len(df), df["sequence_id"].nunique(), dataset_path)

    feature_columns = get_feature_columns(df)
    X, y, y_ord = prepare_features(df, feature_columns)

    leakage_audit(df, feature_columns)

    train_mask, val_mask, test_mask = df["split"] == "train", df["split"] == "val", df["split"] == "test"
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\nSplit sizes: train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    print(f"Sequences: train={df[train_mask]['sequence_id'].nunique()} "
          f"val={df[val_mask]['sequence_id'].nunique()} test={df[test_mask]['sequence_id'].nunique()}")

    class_counts = df.groupby("split")["label"].value_counts().unstack(fill_value=0).reindex(columns=LABELS_ORDERED)
    print("\nClass counts per split:")
    print(class_counts.reindex(["train", "val", "test"]))

    output_dir = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = []

    # --- Logistic Regression: minimal C sweep ---
    for C in (0.1, 1.0, 10.0):
        pipe = build_lr_pipeline(C)
        model = RiskModel.fit(pipe, X_train, y_train, model_version=f"h1{args.version_tag}-logreg")
        val_metrics = model.evaluate(X_val, y_val)
        tmp_path = output_dir / f"_tmp_lr_C{C}.pkl"
        model.save(tmp_path)
        candidates.append({
            "name": f"LogisticRegression(C={C})", "family": "logreg", "model": model,
            "val_metrics": val_metrics, "latency": measure_inference_latency(model, X_val),
            "size_kb": model_size_kb(tmp_path),
        })
        cleanup_tmp_artifact(tmp_path)

    # --- HistGradientBoostingClassifier: minimal complexity sweep ---
    for max_leaf_nodes in (15, 31):
        pipe = build_hgb_pipeline(max_leaf_nodes)
        model = RiskModel.fit(pipe, X_train, y_train, model_version=f"h1{args.version_tag}-hgb")
        val_metrics = model.evaluate(X_val, y_val)
        tmp_path = output_dir / f"_tmp_hgb_{max_leaf_nodes}.pkl"
        model.save(tmp_path)
        candidates.append({
            "name": f"HistGradientBoosting(max_leaf_nodes={max_leaf_nodes})", "family": "hgb", "model": model,
            "val_metrics": val_metrics, "latency": measure_inference_latency(model, X_val),
            "size_kb": model_size_kb(tmp_path),
        })
        cleanup_tmp_artifact(tmp_path)

    print("\n=== Validation comparison (all candidates) ===")
    for c in candidates:
        m = c["val_metrics"]
        print(f"  {c['name']:<40} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
              f"dangerous_recall={m['dangerous_recall']:.4f} false_safe={m['false_safe_rate']:.4f} "
              f"latency={c['latency']['single_sample_latency_ms']:.4f}ms size={c['size_kb']}KB")

    best_logreg = sorted([c for c in candidates if c["family"] == "logreg"],
                          key=lambda c: (-c["val_metrics"]["dangerous_recall"], -c["val_metrics"]["macro_f1"]))[0]
    best_hgb = sorted([c for c in candidates if c["family"] == "hgb"],
                       key=lambda c: (-c["val_metrics"]["dangerous_recall"], -c["val_metrics"]["macro_f1"]))[0]

    selected = select_best([best_logreg, best_hgb])
    print(f"\nSelected model: {selected['name']}")

    # --- Final, single evaluation on the untouched test set ---
    test_metrics = selected["model"].evaluate(X_test, y_test)
    test_latency = measure_inference_latency(selected["model"], X_test)
    print("\n=== TEST SET metrics (evaluated once, selected model only) ===")
    for k, v in test_metrics.items():
        if k not in ("confusion_matrix", "confusion_matrix_labels", "per_class_precision", "per_class_recall"):
            print(f"  {k}: {v}")
    print(f"  per_class_precision: {test_metrics['per_class_precision']}")
    print(f"  per_class_recall: {test_metrics['per_class_recall']}")
    print(f"  confusion_matrix ({LABELS_ORDERED}):")
    for row in test_metrics["confusion_matrix"]:
        print(f"    {row}")

    # --- Feature importance ---
    print("\n=== Top 10 feature importances ===")
    if selected["family"] == "logreg":
        clf = selected["model"].pipeline.named_steps["clf"]
        importance = np.abs(clf.coef_).sum(axis=0)
        importance_pairs = sorted(zip(selected["model"].feature_names, importance), key=lambda p: -p[1])[:10]
        importance_method = "sum(|coefficient|) across classes"
    else:
        result = permutation_importance(
            selected["model"].pipeline, X_val, y_val, n_repeats=5, random_state=42, scoring="f1_macro"
        )
        importance_pairs = sorted(
            zip(selected["model"].feature_names, result.importances_mean), key=lambda p: -p[1]
        )[:10]
        importance_method = "permutation importance (macro F1 drop), 5 repeats on val"

    for name, score in importance_pairs:
        print(f"  {name:<35} {score:.5f}")

    # --- Save the selected model + full report ---
    model_path = output_dir / "risk_model.pkl"
    selected["model"].save(model_path, extra_metadata={
        "dataset_version": args.dataset_version, "feature_schema_version": FEATURE_SCHEMA_VERSION, "training_seed": 42,
        "training_dataset": str(dataset_path), "training_rows": int(len(X_train)), "model_family": selected["name"],
        "training_nan_patterns": nan_pattern_support(X_train), "horizon": "current", "horizon_seconds": 0.0,
    })

    report = {
        "model_version": selected["model"].model_version,
        "selected_model": selected["name"],
        "selection_criteria_order": list(SELECTION_METRICS),
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "split_sequence_counts": {
            "train": int(df[train_mask]["sequence_id"].nunique()),
            "val": int(df[val_mask]["sequence_id"].nunique()),
            "test": int(df[test_mask]["sequence_id"].nunique()),
        },
        "validation_comparison": [
            {"name": c["name"], "val_metrics": c["val_metrics"], "latency": c["latency"], "size_kb": c["size_kb"]}
            for c in candidates
        ],
        "test_metrics": test_metrics,
        "test_latency": test_latency,
        "feature_importance_method": importance_method,
        "top_10_features": [{"feature": n, "importance": float(s)} for n, s in importance_pairs],
        "model_size_kb": model_size_kb(model_path),
        "dangerous_labels": sorted(DANGEROUS_LABELS),
        "disclaimer": (
            "This model is trained ONLY on synthetic data from Phase G1. It demonstrates that the "
            "engineered feature space can support risk-state classification on synthetic scenarios. "
            "It does NOT demonstrate real-world crowd-safety accuracy and is NOT validated for deployment. "
            "The 0-100 risk score is a display/decision-support value, not a probability of injury, "
            "stampede, or any physically validated danger measure."
        ),
    }
    report_path = output_dir / "model_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nModel saved to: {model_path} ({report['model_size_kb']} KB)")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
