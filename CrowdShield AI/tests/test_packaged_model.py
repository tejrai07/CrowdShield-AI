"""Sanity test for the ACTUAL packaged model artifact
(models/artifacts/risk_model.pkl) produced by scripts/train_risk_model.py
— distinct from the other H1 tests, which train small throwaway models on
the fly. Skips gracefully if the artifact hasn't been generated yet.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.synthetic_generator import load_dataset
from models.features import LABELS_ORDERED, prepare_features
from models.risk_model import RiskModel

REPO_ROOT = Path(__file__).parent.parent
MODEL_PATH = REPO_ROOT / "models" / "artifacts" / "risk_model.pkl"
DATASET_PATH = REPO_ROOT / "sample_data" / "synthetic" / "synthetic_dataset.csv"


@pytest.fixture(scope="module")
def packaged_model() -> RiskModel:
    if not MODEL_PATH.exists():
        pytest.skip(f"Packaged model not present at {MODEL_PATH} — run scripts/train_risk_model.py first.")
    return RiskModel.load(MODEL_PATH)


@pytest.fixture(scope="module")
def test_split_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        pytest.skip(f"Packaged dataset not present at {DATASET_PATH}.")
    df = load_dataset(DATASET_PATH)
    return df[df["split"] == "test"]


def test_packaged_model_loads(packaged_model: RiskModel):
    assert packaged_model.feature_names
    assert packaged_model.model_version


def test_packaged_model_predicts_on_test_split(packaged_model: RiskModel, test_split_data: pd.DataFrame):
    X, y, _ = prepare_features(test_split_data, feature_columns=packaged_model.feature_names)
    preds = packaged_model.predict(X.head(50))
    assert set(preds).issubset(set(LABELS_ORDERED))


def test_packaged_model_risk_scores_bounded(packaged_model: RiskModel, test_split_data: pd.DataFrame):
    X, y, _ = prepare_features(test_split_data, feature_columns=packaged_model.feature_names)
    scores = packaged_model.predict_risk_score(X)
    assert (scores >= 0.0).all()
    assert (scores <= 100.0).all()


def test_packaged_model_feature_names_exclude_metadata(packaged_model: RiskModel):
    from models.features import EXCLUDED_COLUMNS

    leaked = EXCLUDED_COLUMNS & set(packaged_model.feature_names)
    assert leaked == set(), f"LEAKAGE in packaged model's feature_names: {leaked}"


def test_packaged_model_has_metadata_sidecar_matching_installed_sklearn():
    """Regression test for the H1 portability fix: the shipped artifact's
    metadata sidecar must record a scikit-learn version that matches the
    pinned requirements.txt version — otherwise a fresh install elsewhere
    would reproduce the original cross-machine failure."""
    import json

    import sklearn

    meta_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".meta.json")
    if not meta_path.exists():
        pytest.skip(f"Metadata sidecar not present at {meta_path}.")

    with meta_path.open() as f:
        metadata = json.load(f)

    assert metadata["sklearn_version"] == sklearn.__version__, (
        f"Packaged model was trained with sklearn {metadata['sklearn_version']} but this "
        f"environment has {sklearn.__version__} installed — requirements.txt's exact pin "
        f"and the shipped artifact have drifted out of sync."
    )
