"""J1 demo: load the packaged H1/I1 models and print a few representative
explanations (current, +5m, +10m) — demonstrates the explainability layer
without generating a large artifact.

Usage:
    python scripts/run_explainability_demo.py \
        --dataset sample_data/synthetic_long/synthetic_dataset.csv \
        --n-examples 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.synthetic_generator import load_dataset  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from models.explainability import MultiHorizonExplainer, format_explanation  # noqa: E402
from models.features import prepare_features  # noqa: E402
from models.future_risk import MultiHorizonRiskModel, prepare_future_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="CrowdShield J1 explainability demo")
    parser.add_argument("--dataset", default="sample_data/synthetic_long/synthetic_dataset.csv")
    parser.add_argument("--artifacts-dir", default="models/artifacts")
    parser.add_argument("--n-examples", type=int, default=3)
    args = parser.parse_args()

    logger = setup_logging("INFO")

    artifacts_dir = REPO_ROOT / args.artifacts_dir if not Path(args.artifacts_dir).is_absolute() else Path(args.artifacts_dir)
    dataset_path = REPO_ROOT / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)

    logger.info("Loading packaged models from %s", artifacts_dir)
    bundle = MultiHorizonRiskModel.load(artifacts_dir)

    logger.info("Loading dataset from %s", dataset_path)
    df = load_dataset(dataset_path)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    X_train, _, _ = prepare_features(train_df, feature_columns=bundle.current.feature_names)
    explainer = MultiHorizonExplainer.from_bundle(bundle, X_train)

    X_test, _, _ = prepare_features(test_df, feature_columns=bundle.current.feature_names)

    # Latency benchmark: prediction alone vs explanation alone vs combined.
    row = X_test.iloc[[0]]
    for _ in range(5):  # warm-up
        bundle.current.predict_risk_score(row)
        explainer.current.explain(row)

    n_bench = 100
    t0 = time.perf_counter()
    for _ in range(n_bench):
        bundle.current.predict_risk_score(row)
    predict_ms = (time.perf_counter() - t0) / n_bench * 1000

    t0 = time.perf_counter()
    for _ in range(n_bench):
        explainer.current.explain(row)
    explain_ms = (time.perf_counter() - t0) / n_bench * 1000

    print("\n=== Latency ===")
    print(f"  prediction only:            {predict_ms:.3f} ms")
    print(f"  explanation (predict+explain): {explain_ms:.3f} ms")
    print(f"  explanation-only overhead:   {explain_ms - predict_ms:.3f} ms")

    print(f"\n=== {args.n_examples} representative explanations ===")
    for i in range(min(args.n_examples, len(X_test))):
        example_row = X_test.iloc[[i]]
        print(f"\n{'=' * 60}")
        print(f"Example {i + 1} (test row index {X_test.index[i]})")
        print("=" * 60)

        results = explainer.explain_all(example_row)
        for horizon_key in ("current", "future_5m", "future_10m"):
            print()
            print(format_explanation(results[horizon_key]))

    print(
        "\nNOTE: +5m/+10m explanations describe which CURRENT features drove "
        "the model's forecast. I1's own evaluation found escalation_recall "
        "of only ~12.5% (5m) / ~10.7% (10m) — these future explanations "
        "should NOT be read as a reliable early-warning signal. See README "
        "'Phase I1' and 'Phase J1' for the full disclaimer."
    )


if __name__ == "__main__":
    main()
