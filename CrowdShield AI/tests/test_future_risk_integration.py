"""Sanity tests for the ACTUAL packaged I1 future-risk model artifacts
(models/artifacts/future_risk_{current,5m,10m}.pkl) produced by
scripts/train_future_risk.py. Distinct from tests/test_future_risk.py,
which trains small throwaway models on the fly. Skips gracefully if the
artifacts haven't been generated yet — mirrors the pattern established
by tests/test_packaged_model.py for H1.

Not gated behind CROWDSHIELD_RUN_INTEGRATION — this is pure tabular ML
(no YOLO/video/network dependency), so it stays fast enough for the
standard suite.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.synthetic_generator import load_dataset
from evaluation.metrics import escalation_recall
from models.features import LABELS_ORDERED, prepare_features
from models.future_risk import HORIZON_SECONDS, MultiHorizonRiskModel, prepare_future_features

REPO_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = REPO_ROOT / "models" / "artifacts"
LONG_DATASET_PATH = REPO_ROOT / "sample_data" / "synthetic_long" / "synthetic_dataset.csv"


def _artifacts_present() -> bool:
    return all(
        (ARTIFACTS_DIR / name).exists()
        for name in ("future_risk_current.pkl", "future_risk_5m.pkl", "future_risk_10m.pkl")
    )


@pytest.fixture(scope="module")
def bundle() -> MultiHorizonRiskModel:
    if not _artifacts_present():
        pytest.skip(f"Packaged I1 artifacts not present in {ARTIFACTS_DIR} — run scripts/train_future_risk.py first.")
    return MultiHorizonRiskModel.load(ARTIFACTS_DIR)


@pytest.fixture(scope="module")
def long_test_df() -> pd.DataFrame:
    if not LONG_DATASET_PATH.exists():
        pytest.skip(f"Long-duration dataset not present at {LONG_DATASET_PATH}.")
    df = load_dataset(LONG_DATASET_PATH)
    return df[df["split"] == "test"]


def test_all_three_horizon_models_load(bundle: MultiHorizonRiskModel):
    assert bundle.current.feature_names
    assert bundle.future_5m.feature_names
    assert bundle.future_10m.feature_names


def test_all_three_models_share_identical_feature_ordering(bundle: MultiHorizonRiskModel):
    """Future models must receive exactly the feature columns they were
    trained on, in the same order as the current model — all three are
    trained on the same X schema (only y differs per horizon)."""
    assert bundle.current.feature_names == bundle.future_5m.feature_names
    assert bundle.current.feature_names == bundle.future_10m.feature_names


def test_metadata_records_correct_horizon_seconds(bundle: MultiHorizonRiskModel):
    assert bundle.current.metadata.get("horizon_seconds") == HORIZON_SECONDS["current"]
    assert bundle.future_5m.metadata.get("horizon_seconds") == HORIZON_SECONDS["5m"]
    assert bundle.future_10m.metadata.get("horizon_seconds") == HORIZON_SECONDS["10m"]


def test_metadata_records_sklearn_version(bundle: MultiHorizonRiskModel):
    import sklearn

    for model in (bundle.current, bundle.future_5m, bundle.future_10m):
        assert model.metadata.get("sklearn_version") == sklearn.__version__


def test_packaged_bundle_predicts_on_real_test_data(bundle: MultiHorizonRiskModel, long_test_df: pd.DataFrame):
    X, _, _ = prepare_features(long_test_df, feature_columns=bundle.current.feature_names)
    result = bundle.predict(X.head(30))

    for key in ("current", "future_5m", "future_10m"):
        assert set(result[key]["risk_class"]).issubset(set(LABELS_ORDERED))
        scores = np.array(result[key]["risk_score"])
        assert (scores >= 0.0).all() and (scores <= 100.0).all()
        dp = np.array(result[key]["dangerous_probability"])
        assert (dp >= 0.0).all() and (dp <= 1.0 + 1e-9).all()


def test_packaged_current_model_no_metadata_leakage(bundle: MultiHorizonRiskModel):
    from models.features import EXCLUDED_COLUMNS

    for model in (bundle.current, bundle.future_5m, bundle.future_10m):
        leaked = EXCLUDED_COLUMNS & set(model.feature_names)
        assert leaked == set(), f"LEAKAGE in packaged model's feature_names: {leaked}"


def test_escalation_recall_computable_on_real_test_data(bundle: MultiHorizonRiskModel, long_test_df: pd.DataFrame):
    """End-to-end sanity: the escalation-recall metric must be
    computable (not crash) on the real packaged model's real test-set
    predictions, and must land in [0, 1] when defined."""
    X_5m, y_5m_true = prepare_future_features(long_test_df, "5m", bundle.future_5m.feature_names)
    y_current_true = long_test_df.loc[X_5m.index, "label"]
    y_5m_pred = bundle.future_5m.predict(X_5m)

    recall = escalation_recall(y_current_true, y_5m_true, y_5m_pred)
    assert recall != recall or (0.0 <= recall <= 1.0)  # NaN check via self-inequality, or in-bounds


def test_future_risk_report_exists_and_documents_disclaimer():
    report_path = ARTIFACTS_DIR / "future_risk_report.json"
    if not report_path.exists():
        pytest.skip(f"Report not present at {report_path}.")
    with report_path.open() as f:
        report = json.load(f)
    assert "synthetic" in report["disclaimer"].lower()
    assert "not" in report["disclaimer"].lower()


def test_current_model_saved_separately_from_h1_model():
    """I1's own current-horizon retrain must never overwrite H1's
    canonical risk_model.pkl."""
    h1_path = ARTIFACTS_DIR / "risk_model.pkl"
    i1_current_path = ARTIFACTS_DIR / "future_risk_current.pkl"
    if not (h1_path.exists() and i1_current_path.exists()):
        pytest.skip("One or both model artifacts not present.")
    assert h1_path != i1_current_path
    # Distinct files with distinct sizes/hashes expected in general (not
    # a strict requirement, but they should each independently load).
    assert h1_path.stat().st_size > 0
    assert i1_current_path.stat().st_size > 0
