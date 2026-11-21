"""Unit tests for models/risk_model.py and models/features.py.

Trains tiny, fast models on a small synthetic dataset generated on the
fly (via G1's own generator, not a fixed large fixture) — keeps this part
of the standard fast test suite while still exercising the real
fit/predict/save/load path end-to-end.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.schemas import GeneratorConfig
from data.synthetic_generator import generate_dataset
from models.features import (
    DANGEROUS_LABELS,
    EXCLUDED_COLUMNS,
    EXCLUDED_NON_PREDICTIVE,
    LABELS_ORDERED,
    ORDINAL_MAP,
    get_feature_columns,
    prepare_features,
)
from models.risk_model import RiskModel


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    from data.loader import load_generator_config

    base_cfg = load_generator_config()
    tiny_cfg = base_cfg.model_copy(
        update={"realizations_per_scenario": 2, "realizations_per_transition": 1, "duration_s": 30.0}
    )
    df, _ = generate_dataset(tiny_cfg)
    return df


@pytest.fixture(scope="module")
def fitted_model(small_df: pd.DataFrame) -> RiskModel:
    X, y, _ = prepare_features(small_df)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    return RiskModel.fit(pipe, X, y)


# --- feature selection / leakage ------------------------------------------------

def test_feature_columns_exclude_metadata_and_targets(small_df: pd.DataFrame):
    feature_columns = get_feature_columns(small_df)
    leaked = EXCLUDED_COLUMNS & set(feature_columns)
    assert leaked == set(), f"leaked columns: {leaked}"


def test_feature_columns_exclude_raw_direction(small_df: pd.DataFrame):
    feature_columns = get_feature_columns(small_df)
    assert "flow_direction_deg" not in feature_columns
    assert EXCLUDED_NON_PREDICTIVE.issubset({"flow_direction_deg"})


def test_prepare_features_bool_columns_cast_to_int(small_df: pd.DataFrame):
    X, y, y_ord = prepare_features(small_df)
    assert X["surge_flag"].dtype != bool
    assert set(X["surge_flag"].unique()).issubset({0, 1})


def test_prepare_features_ordinal_mapping_correct(small_df: pd.DataFrame):
    X, y, y_ord = prepare_features(small_df)
    for label, ordinal in ORDINAL_MAP.items():
        matching = y_ord[y == label]
        if len(matching) > 0:
            assert (matching == ordinal).all()


def test_ordinal_map_is_stable_order():
    assert ORDINAL_MAP == {"SAFE": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4}
    assert LABELS_ORDERED == ["SAFE", "LOW", "MODERATE", "HIGH", "CRITICAL"]


def test_dangerous_labels_are_high_and_critical_only():
    assert DANGEROUS_LABELS == {"HIGH", "CRITICAL"}


# --- training / prediction --------------------------------------------------------

def test_model_trains_successfully(fitted_model: RiskModel):
    assert fitted_model.feature_names
    assert hasattr(fitted_model.pipeline, "classes_")


def test_predict_returns_valid_labels(fitted_model: RiskModel, small_df: pd.DataFrame):
    X, y, _ = prepare_features(small_df)
    preds = fitted_model.predict(X.head(10))
    assert set(preds).issubset(set(LABELS_ORDERED))
    assert len(preds) == 10


def test_predict_proba_columns_in_ordinal_order(fitted_model: RiskModel, small_df: pd.DataFrame):
    X, y, _ = prepare_features(small_df)
    proba = fitted_model.predict_proba(X.head(5))
    assert list(proba.columns) == LABELS_ORDERED


def test_predict_proba_handles_class_missing_from_training_data(small_df: pd.DataFrame):
    """Regression test for a real bug caught during J1 testing: if a
    train split happens to be missing one class entirely, sklearn's
    pipeline.classes_ only has 4 classes, and predict_proba() must still
    return all 5 LABELS_ORDERED columns (0.0 for the unseen class) rather
    than raising a KeyError."""
    X, y, _ = prepare_features(small_df)

    # Deliberately drop one class from the training labels to force the
    # exact condition that caused the original crash.
    keep_mask = y != "MODERATE"
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    model = RiskModel.fit(pipe, X[keep_mask], y[keep_mask])

    proba = model.predict_proba(X.head(5))
    assert list(proba.columns) == LABELS_ORDERED
    assert (proba["MODERATE"] == 0.0).all()
    assert np.allclose(proba.sum(axis=1).values, 1.0, atol=1e-6)


def test_predict_proba_sums_to_one(fitted_model: RiskModel, small_df: pd.DataFrame):
    X, y, _ = prepare_features(small_df)
    proba = fitted_model.predict_proba(X.head(20))
    sums = proba.sum(axis=1)
    assert np.allclose(sums.values, 1.0, atol=1e-6)


def test_risk_score_bounded_0_to_100(fitted_model: RiskModel, small_df: pd.DataFrame):
    X, y, _ = prepare_features(small_df)
    scores = fitted_model.predict_risk_score(X)
    assert (scores >= 0.0).all()
    assert (scores <= 100.0).all()


def test_risk_score_higher_for_confident_critical_prediction(fitted_model: RiskModel):
    """A hand-built all-CRITICAL-probability row must score near 100;
    an all-SAFE-probability row must score near 0. Tests the formula
    directly rather than relying on the model's own predictions."""
    # Bypass the model — verify the arithmetic via a stub with fixed proba.
    from models.risk_model import _ORDINAL_WEIGHTS

    critical_proba = np.zeros(5)
    critical_proba[LABELS_ORDERED.index("CRITICAL")] = 1.0
    safe_proba = np.zeros(5)
    safe_proba[LABELS_ORDERED.index("SAFE")] = 1.0

    assert 100.0 * (critical_proba @ _ORDINAL_WEIGHTS) == pytest.approx(100.0)
    assert 100.0 * (safe_proba @ _ORDINAL_WEIGHTS) == pytest.approx(0.0)


def test_missing_feature_raises_clear_error(fitted_model: RiskModel, small_df: pd.DataFrame):
    X, y, _ = prepare_features(small_df)
    X_missing = X.drop(columns=[fitted_model.feature_names[0]])
    with pytest.raises(ValueError):
        fitted_model.predict(X_missing)


# --- serialization -----------------------------------------------------------------

def test_save_load_roundtrip_identical_predictions(fitted_model: RiskModel, small_df: pd.DataFrame, tmp_path):
    X, y, _ = prepare_features(small_df)
    X_sample = X.head(15)

    original_preds = fitted_model.predict(X_sample)
    original_proba = fitted_model.predict_proba(X_sample)
    original_risk = fitted_model.predict_risk_score(X_sample)

    save_path = tmp_path / "test_model.pkl"
    fitted_model.save(save_path)
    assert save_path.exists()

    loaded = RiskModel.load(save_path)
    loaded_preds = loaded.predict(X_sample)
    loaded_proba = loaded.predict_proba(X_sample)
    loaded_risk = loaded.predict_risk_score(X_sample)

    assert list(original_preds) == list(loaded_preds)
    pd.testing.assert_frame_equal(original_proba, loaded_proba)
    assert np.allclose(original_risk, loaded_risk)
    assert loaded.feature_names == fitted_model.feature_names
    assert loaded.model_version == fitted_model.model_version


# --- portability fix (see README "H1 portability fix") ---------------------------

def test_save_writes_metadata_sidecar_with_sklearn_version(fitted_model: RiskModel, tmp_path):
    """save() must write a plain-JSON metadata sidecar recording the
    training-time scikit-learn version — this is what let a real
    cross-machine ModuleNotFoundError be turned into a clear, actionable
    error instead of a bare traceback."""
    import json

    import sklearn

    save_path = tmp_path / "portability_test.pkl"
    fitted_model.save(save_path)

    meta_path = tmp_path / "portability_test.pkl.meta.json"
    assert meta_path.exists()

    with meta_path.open() as f:
        metadata = json.load(f)

    assert metadata["sklearn_version"] == sklearn.__version__
    assert "feature_names" in metadata
    assert "model_version" in metadata


def test_load_missing_metadata_sidecar_raises_clear_error(fitted_model: RiskModel, tmp_path):
    """If the pipeline file exists but its metadata sidecar doesn't
    (e.g. a partial/corrupted copy), load() must fail with a clear,
    actionable FileNotFoundError, not a confusing downstream error."""
    save_path = tmp_path / "no_sidecar.pkl"
    fitted_model.save(save_path)
    (tmp_path / "no_sidecar.pkl.meta.json").unlink()  # simulate a missing sidecar

    with pytest.raises(FileNotFoundError, match="metadata sidecar"):
        RiskModel.load(save_path)


def test_load_warns_on_sklearn_version_mismatch(fitted_model: RiskModel, tmp_path):
    """Regression test for the real cross-machine portability failure:
    when the recorded training-time sklearn version differs from the
    currently installed one, load() must emit a RuntimeWarning naming
    both versions and the fix — this is the mechanism that turns a
    cryptic ModuleNotFoundError('_loss') into an actionable message.

    NOTE: this sandbox only has one scikit-learn version installed, so we
    cannot literally reproduce the original cross-version LOAD FAILURE
    here (that would require two different sklearn installations, e.g. a
    genuine Windows machine on a different version — see README for the
    explicit statement that this is not fabricated). This test instead
    verifies the DETECTION mechanism: a tampered metadata file with a
    different recorded version reliably triggers the warning.
    """
    save_path = tmp_path / "version_mismatch.pkl"
    fitted_model.save(save_path)

    meta_path = tmp_path / "version_mismatch.pkl.meta.json"
    with meta_path.open() as f:
        metadata = json.load(f)
    metadata["sklearn_version"] = "0.0.0-fake-old-version"
    with meta_path.open("w") as f:
        json.dump(metadata, f)

    with pytest.warns(RuntimeWarning, match="0.0.0-fake-old-version"):
        loaded = RiskModel.load(save_path)

    # The installed sklearn version is actually unchanged in this sandbox,
    # so the underlying joblib.load() itself still succeeds — only the
    # metadata's recorded version was tampered with. This confirms
    # detection fires correctly even though we can't force an actual
    # cross-version load failure in a single-Python-environment sandbox.
    assert loaded.feature_names == fitted_model.feature_names


def test_load_wraps_actual_load_failure_with_clear_message(fitted_model: RiskModel, tmp_path):
    """If joblib.load() itself fails (simulating what a genuine
    cross-version incompatibility does), load() must re-raise as a
    RuntimeError whose message names both sklearn versions and the exact
    remediation command — not propagate the raw underlying exception."""
    save_path = tmp_path / "corrupted.pkl"
    fitted_model.save(save_path)

    # Corrupt the pipeline file's bytes to force an actual load failure —
    # the closest we can get to simulating "unpickling failed" without a
    # second sklearn installation.
    with save_path.open("r+b") as f:
        f.write(b"not a valid joblib/pickle file at all")

    with pytest.raises(RuntimeError, match="pip install scikit-learn=="):
        RiskModel.load(save_path)


# --- determinism ---------------------------------------------------------------------

def test_same_config_produces_reproducible_predictions(small_df: pd.DataFrame):
    X, y, _ = prepare_features(small_df)

    def _train():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ])
        return RiskModel.fit(pipe, X, y)

    model_a = _train()
    model_b = _train()

    preds_a = model_a.predict(X.head(20))
    preds_b = model_b.predict(X.head(20))
    assert list(preds_a) == list(preds_b)
