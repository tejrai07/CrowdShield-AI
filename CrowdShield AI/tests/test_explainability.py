"""Unit tests for models/explainability.py.

Trains a small, fast model on the fly (via G1's own generator) — keeps
this part of the standard fast test suite while exercising the real
explain() path end-to-end, including current and future horizons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.loader import load_generator_config
from data.synthetic_generator import generate_dataset, split_dataset
from models.explainability import (
    FeatureContribution,
    ModelExplainer,
    MultiHorizonExplainer,
    RiskExplanation,
    format_explanation,
    human_readable_feature_name,
)
from models.features import LABELS_ORDERED, get_feature_columns, prepare_features
from models.future_risk import MultiHorizonRiskModel, prepare_future_features
from models.risk_model import RiskModel
from models.training_utils import build_hgb_pipeline


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    base_cfg = load_generator_config()
    cfg = base_cfg.model_validate({
        **base_cfg.model_dump(),
        "duration_s": 700.0, "realizations_per_scenario": 1, "realizations_per_transition": 1,
    })
    df, _ = generate_dataset(cfg)
    return split_dataset(df, seed=1, train_frac=0.7, val_frac=0.15, test_frac=0.15)


@pytest.fixture(scope="module")
def feature_columns(small_df: pd.DataFrame) -> list[str]:
    return get_feature_columns(small_df)


@pytest.fixture(scope="module")
def fitted_model(small_df: pd.DataFrame, feature_columns: list[str]) -> RiskModel:
    X, y, _ = prepare_features(small_df, feature_columns)
    train_mask = small_df.loc[X.index, "split"] == "train"
    pipe = build_hgb_pipeline(15)
    return RiskModel.fit(pipe, X[train_mask], y[train_mask], model_version="test-explain")


@pytest.fixture(scope="module")
def explainer(fitted_model: RiskModel, small_df: pd.DataFrame, feature_columns: list[str]) -> ModelExplainer:
    X, y, _ = prepare_features(small_df, feature_columns)
    train_mask = small_df.loc[X.index, "split"] == "train"
    return ModelExplainer.from_training_data(fitted_model, X[train_mask], horizon="current", horizon_seconds=0.0)


@pytest.fixture(scope="module")
def test_X(small_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    X, y, _ = prepare_features(small_df, feature_columns)
    test_mask = small_df.loc[X.index, "split"] == "test"
    return X[test_mask]


# --- explanation object creation --------------------------------------------------

def test_explain_returns_risk_explanation(explainer: ModelExplainer, test_X: pd.DataFrame):
    explanation = explainer.explain(test_X.iloc[[0]])
    assert isinstance(explanation, RiskExplanation)
    assert explanation.predicted_class in LABELS_ORDERED


def test_explain_requires_exactly_one_row(explainer: ModelExplainer, test_X: pd.DataFrame):
    with pytest.raises(ValueError):
        explainer.explain(test_X.head(2))
    with pytest.raises(ValueError):
        explainer.explain(test_X.head(0))


# --- feature names match trained model ------------------------------------------

def test_feature_contributions_cover_all_trained_features(explainer: ModelExplainer, fitted_model: RiskModel, test_X: pd.DataFrame):
    explanation = explainer.explain(test_X.iloc[[0]])
    assert set(explanation.feature_contributions.keys()) == set(fitted_model.feature_names)
    assert set(explanation.feature_values.keys()) == set(fitted_model.feature_names)


def test_top_features_are_a_subset_of_trained_features(explainer: ModelExplainer, fitted_model: RiskModel, test_X: pd.DataFrame):
    explanation = explainer.explain(test_X.iloc[[0]])
    all_top_feature_names = {fc.feature for fc in explanation.top_positive_features + explanation.top_negative_features}
    assert all_top_feature_names.issubset(set(fitted_model.feature_names))


def test_no_metadata_columns_ever_used_as_features(explainer: ModelExplainer, test_X: pd.DataFrame):
    from models.features import EXCLUDED_COLUMNS

    explanation = explainer.explain(test_X.iloc[[0]])
    leaked = EXCLUDED_COLUMNS & set(explanation.feature_contributions.keys())
    assert leaked == set(), f"LEAKAGE: {leaked} found in explanation features"


# --- dimensions ------------------------------------------------------------------

def test_top_k_respected(explainer: ModelExplainer, test_X: pd.DataFrame):
    explanation = explainer.explain(test_X.iloc[[0]], top_k=3)
    assert len(explanation.top_positive_features) <= 3
    assert len(explanation.top_negative_features) <= 3


def test_positive_features_sorted_descending(explainer: ModelExplainer, test_X: pd.DataFrame):
    explanation = explainer.explain(test_X.iloc[[0]])
    contributions = [fc.contribution for fc in explanation.top_positive_features]
    assert contributions == sorted(contributions, reverse=True)
    assert all(c > 0 for c in contributions)


def test_negative_features_sorted_ascending(explainer: ModelExplainer, test_X: pd.DataFrame):
    explanation = explainer.explain(test_X.iloc[[0]])
    contributions = [fc.contribution for fc in explanation.top_negative_features]
    assert contributions == sorted(contributions)
    assert all(c < 0 for c in contributions)


# --- determinism ------------------------------------------------------------------

def test_deterministic_for_same_input(explainer: ModelExplainer, test_X: pd.DataFrame):
    row = test_X.iloc[[0]]
    e1 = explainer.explain(row)
    e2 = explainer.explain(row)
    assert e1.predicted_class == e2.predicted_class
    assert e1.feature_contributions == e2.feature_contributions


# --- predicted class/probability match RiskModel ---------------------------------

def test_predicted_class_matches_risk_model(explainer: ModelExplainer, fitted_model: RiskModel, test_X: pd.DataFrame):
    for i in range(5):
        row = test_X.iloc[[i]]
        explanation = explainer.explain(row)
        assert explanation.predicted_class == fitted_model.predict(row)[0]


def test_predicted_probability_matches_risk_model(explainer: ModelExplainer, fitted_model: RiskModel, test_X: pd.DataFrame):
    row = test_X.iloc[[0]]
    explanation = explainer.explain(row)
    proba = fitted_model.predict_proba(row)
    expected = proba[explanation.predicted_class].iloc[0]
    assert explanation.predicted_probability == pytest.approx(expected)


def test_risk_score_matches_risk_model(explainer: ModelExplainer, fitted_model: RiskModel, test_X: pd.DataFrame):
    row = test_X.iloc[[0]]
    explanation = explainer.explain(row)
    expected = fitted_model.predict_risk_score(row)[0]
    assert explanation.risk_score == pytest.approx(expected)


# --- NaN / missing feature handling -----------------------------------------------

def test_nan_feature_value_does_not_crash(explainer: ModelExplainer, test_X: pd.DataFrame):
    """Rows early in a sequence have NaN slope/rate features (F1 warm-up
    convention) — explain() must handle this gracefully, not crash."""
    nan_rows = test_X[test_X.isna().any(axis=1)]
    if len(nan_rows) == 0:
        pytest.skip("No NaN rows in this small test dataset to exercise this case.")
    explanation = explainer.explain(nan_rows.iloc[[0]])
    assert explanation.predicted_class in LABELS_ORDERED


def test_format_explanation_handles_nan_gracefully():
    fc_nan = FeatureContribution(
        feature="w5_density_slope", human_name="5s density trend",
        value=float("nan"), reference_value=0.02, contribution=-0.1,
    )
    explanation = RiskExplanation(
        predicted_class="SAFE", predicted_probability=0.9, risk_score=10.0,
        horizon="current", horizon_seconds=0.0, model_version="test",
        top_positive_features=[], top_negative_features=[fc_nan],
        feature_contributions={"w5_density_slope": -0.1}, feature_values={"w5_density_slope": float("nan")},
    )
    text = format_explanation(explanation)
    assert "nan" not in text.lower().replace("nano", "")  # no raw "nan" leaking into the text
    assert "not available" in text.lower()


# --- human-readable formatting -----------------------------------------------------

def test_human_readable_feature_name_window_prefixed():
    assert human_readable_feature_name("w30_density_slope") == "30s density trend"
    assert human_readable_feature_name("w60_bottleneck_pressure") == "60s bottleneck pressure"


def test_human_readable_feature_name_base_metric():
    assert human_readable_feature_name("direction_variance") == "movement direction variance (conflicting movement)"
    assert human_readable_feature_name("density") == "crowd density"


def test_human_readable_feature_name_unknown_falls_back_to_raw():
    assert human_readable_feature_name("totally_unknown_feature_xyz") == "totally_unknown_feature_xyz"


def test_format_explanation_avoids_fabricated_psychological_language(explainer: ModelExplainer, test_X: pd.DataFrame):
    explanation = explainer.explain(test_X.iloc[[0]])
    text = format_explanation(explanation)
    forbidden_phrases = ["panic", "scared", "afraid", "terrified", "chaos"]
    for phrase in forbidden_phrases:
        assert phrase not in text.lower()


def test_format_explanation_contains_class_and_score(explainer: ModelExplainer, test_X: pd.DataFrame):
    explanation = explainer.explain(test_X.iloc[[0]])
    text = format_explanation(explanation)
    assert explanation.predicted_class in text
    assert "Risk score" in text


# --- idxmax equivalence (mirrors I1's MultiHorizonRiskModel fix) -----------------

def test_predicted_class_via_idxmax_matches_pipeline_predict(fitted_model: RiskModel, test_X: pd.DataFrame):
    proba = fitted_model.predict_proba(test_X.head(50))
    via_idxmax = list(proba.idxmax(axis=1))
    via_predict = list(fitted_model.predict(test_X.head(50)))
    assert via_idxmax == via_predict


# --- MultiHorizonExplainer (current + 5m + 10m) -----------------------------------

@pytest.fixture(scope="module")
def multi_horizon_setup(small_df: pd.DataFrame, feature_columns: list[str]):
    models = {}
    for horizon in ("current", "5m", "10m"):
        X, y = prepare_future_features(small_df, horizon, feature_columns)
        train_mask = small_df.loc[X.index, "split"] == "train"
        pipe = build_hgb_pipeline(15)
        models[horizon] = RiskModel.fit(pipe, X[train_mask], y[train_mask], model_version=f"test-{horizon}")

    bundle = MultiHorizonRiskModel(current=models["current"], future_5m=models["5m"], future_10m=models["10m"])

    X_current, _ = prepare_future_features(small_df, "current", feature_columns)
    train_mask = small_df.loc[X_current.index, "split"] == "train"
    reference_df = X_current[train_mask]

    multi_explainer = MultiHorizonExplainer.from_bundle(bundle, reference_df)
    return bundle, multi_explainer, X_current


def test_multi_horizon_explain_all_returns_three_horizons(multi_horizon_setup):
    bundle, multi_explainer, X_current = multi_horizon_setup
    results = multi_explainer.explain_all(X_current.iloc[[0]])
    assert set(results.keys()) == {"current", "future_5m", "future_10m"}
    for key in results:
        assert isinstance(results[key], RiskExplanation)


def test_multi_horizon_explanations_have_correct_horizon_seconds(multi_horizon_setup):
    bundle, multi_explainer, X_current = multi_horizon_setup
    results = multi_explainer.explain_all(X_current.iloc[[0]])
    assert results["current"].horizon_seconds == 0.0
    assert results["future_5m"].horizon_seconds == 300.0
    assert results["future_10m"].horizon_seconds == 600.0


def test_current_5m_10m_explanations_are_independent(multi_horizon_setup):
    """Different horizon models can (and often do) predict different
    classes/scores for the same input row — explanations must reflect
    each model's own output, not accidentally share state."""
    bundle, multi_explainer, X_current = multi_horizon_setup
    row = X_current.iloc[[0]]
    results = multi_explainer.explain_all(row)
    assert results["current"].predicted_class == bundle.current.predict(row)[0]
    assert results["future_5m"].predicted_class == bundle.future_5m.predict(row)[0]
    assert results["future_10m"].predicted_class == bundle.future_10m.predict(row)[0]


# --- serialization compatibility --------------------------------------------------

def test_explainer_works_after_model_save_load_roundtrip(fitted_model: RiskModel, small_df, feature_columns, tmp_path):
    X, y, _ = prepare_features(small_df, feature_columns)
    train_mask = small_df.loc[X.index, "split"] == "train"
    test_mask = small_df.loc[X.index, "split"] == "test"

    save_path = tmp_path / "explain_test_model.pkl"
    fitted_model.save(save_path)
    loaded_model = RiskModel.load(save_path)

    loaded_explainer = ModelExplainer.from_training_data(loaded_model, X[train_mask])
    original_explainer = ModelExplainer.from_training_data(fitted_model, X[train_mask])

    row = X[test_mask].iloc[[0]]
    e_loaded = loaded_explainer.explain(row)
    e_original = original_explainer.explain(row)

    assert e_loaded.predicted_class == e_original.predicted_class
    assert e_loaded.feature_contributions == e_original.feature_contributions
