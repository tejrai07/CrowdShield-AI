"""Unit tests for models/future_risk.py.

Uses a small, fast, on-the-fly generated long-duration dataset (via G1's
own generator, not a fixed large fixture) — keeps this part of the
standard fast test suite while still exercising real horizon-truncation
and leakage behavior end-to-end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.loader import load_generator_config
from data.synthetic_generator import generate_dataset, split_dataset
from models.features import LABELS_ORDERED, get_feature_columns
from models.future_risk import (
    HORIZON_COLUMNS,
    HORIZON_SECONDS,
    MultiHorizonRiskModel,
    dangerous_probability,
    prepare_future_features,
)
from models.risk_model import RiskModel
from models.training_utils import build_hgb_pipeline


@pytest.fixture(scope="module")
def small_long_df() -> pd.DataFrame:
    """~700s sequences, pure scenarios only, 1 realization each — enough
    to genuinely exercise 5m/10m horizon truncation (300s/600s < 700s)
    while staying fast (~1-2s to generate)."""
    base_cfg = load_generator_config()
    cfg = base_cfg.model_validate({
        **base_cfg.model_dump(),
        "duration_s": 700.0, "realizations_per_scenario": 1, "realizations_per_transition": 1,
    })
    df, _ = generate_dataset(cfg)
    df = split_dataset(df, seed=1, train_frac=0.7, val_frac=0.15, test_frac=0.15)
    return df


@pytest.fixture(scope="module")
def feature_columns(small_long_df: pd.DataFrame) -> list[str]:
    return get_feature_columns(small_long_df)


@pytest.fixture(scope="module")
def fitted_current_and_5m(small_long_df: pd.DataFrame, feature_columns: list[str]):
    models = {}
    for horizon in ("current", "5m"):
        X, y = prepare_future_features(small_long_df, horizon, feature_columns)
        train_mask = small_long_df.loc[X.index, "split"] == "train"
        pipe = build_hgb_pipeline(15)
        models[horizon] = RiskModel.fit(pipe, X[train_mask], y[train_mask], model_version=f"test-{horizon}")
    return models


# --- A. future label shift correctness ------------------------------------------

def test_future_5min_label_matches_actual_label_300s_later(small_long_df: pd.DataFrame):
    df = small_long_df
    sample = df[df["future_5min_label"].notna()].sample(30, random_state=1)
    for _, row in sample.iterrows():
        target_ts = row["timestamp_s"] + 300.0
        match = df[(df["sequence_id"] == row["sequence_id"]) & (df["timestamp_s"] == target_ts)]
        assert len(match) == 1
        assert match.iloc[0]["label"] == row["future_5min_label"]


def test_future_10min_label_matches_actual_label_600s_later(small_long_df: pd.DataFrame):
    df = small_long_df
    sample = df[df["future_10min_label"].notna()].sample(20, random_state=1)
    for _, row in sample.iterrows():
        target_ts = row["timestamp_s"] + 600.0
        match = df[(df["sequence_id"] == row["sequence_id"]) & (df["timestamp_s"] == target_ts)]
        assert len(match) == 1
        assert match.iloc[0]["label"] == row["future_10min_label"]


# --- B. sequence-boundary leakage -------------------------------------------------

def test_future_label_never_crosses_sequence_boundary(small_long_df: pd.DataFrame):
    """The future label at t must come from the SAME sequence_id — never
    borrowed from a different sequence, even one running at similar
    timestamps."""
    df = small_long_df
    # Pick a row near a sequence's actual end where +300s would only
    # exist in a DIFFERENT sequence's timeline (if leakage occurred, the
    # future label would incorrectly be populated from that other
    # sequence instead of correctly being NaN).
    for sequence_id, group in df.groupby("sequence_id"):
        max_ts = group["timestamp_s"].max()
        near_end = group[group["timestamp_s"] > max_ts - 300]
        # None of these rows should have a non-null future_5min_label,
        # since t+300 exceeds this sequence's own last timestamp.
        assert near_end["future_5min_label"].isna().all(), (
            f"sequence {sequence_id} has a non-null future_5min_label within "
            f"300s of its own end — possible cross-sequence leakage"
        )


# --- C. horizon truncation -----------------------------------------------------

def test_last_300s_has_no_valid_5m_target(small_long_df: pd.DataFrame):
    df = small_long_df
    for sequence_id, group in df.groupby("sequence_id"):
        max_ts = group["timestamp_s"].max()
        last_300s = group[group["timestamp_s"] > max_ts - 300]
        assert last_300s["future_5min_label"].isna().all()


def test_last_600s_has_no_valid_10m_target(small_long_df: pd.DataFrame):
    df = small_long_df
    for sequence_id, group in df.groupby("sequence_id"):
        max_ts = group["timestamp_s"].max()
        last_600s = group[group["timestamp_s"] > max_ts - 600]
        assert last_600s["future_10min_label"].isna().all()


def test_prepare_future_features_drops_truncated_rows(small_long_df: pd.DataFrame, feature_columns):
    X_current, y_current = prepare_future_features(small_long_df, "current", feature_columns)
    X_5m, y_5m = prepare_future_features(small_long_df, "5m", feature_columns)
    X_10m, y_10m = prepare_future_features(small_long_df, "10m", feature_columns)

    assert len(X_current) == len(small_long_df)  # current label always defined
    assert len(X_5m) < len(small_long_df)          # some rows truncated
    assert len(X_10m) < len(X_5m)                   # 10m truncates more than 5m
    assert y_5m.notna().all()
    assert y_10m.notna().all()


# --- D. feature leakage -----------------------------------------------------------

def test_future_labels_never_in_feature_columns(feature_columns):
    assert "future_5min_label" not in feature_columns
    assert "future_10min_label" not in feature_columns
    assert "label" not in feature_columns


def test_prepare_future_features_X_excludes_targets(small_long_df: pd.DataFrame, feature_columns):
    for horizon in ("current", "5m", "10m"):
        X, y = prepare_future_features(small_long_df, horizon, feature_columns)
        assert "label" not in X.columns
        assert "future_5min_label" not in X.columns
        assert "future_10min_label" not in X.columns


def test_unknown_horizon_raises():
    with pytest.raises(ValueError):
        prepare_future_features(pd.DataFrame({"label": ["SAFE"]}), "15m")


# --- E. feature ordering compatibility --------------------------------------------

def test_model_receives_exact_trained_feature_columns(fitted_current_and_5m, small_long_df, feature_columns):
    model = fitted_current_and_5m["current"]
    X, _ = prepare_future_features(small_long_df, "current", feature_columns)
    # Shuffle column order — model must still select its own trained order internally.
    shuffled = X[list(reversed(X.columns))]
    preds_shuffled = model.predict(shuffled.head(5))
    preds_original = model.predict(X.head(5))
    assert list(preds_shuffled) == list(preds_original)


# --- F. serialization round-trip ---------------------------------------------------

def test_future_model_save_load_roundtrip(fitted_current_and_5m, small_long_df, feature_columns, tmp_path):
    model = fitted_current_and_5m["5m"]
    X, _ = prepare_future_features(small_long_df, "5m", feature_columns)
    X_sample = X.head(10)

    original_preds = model.predict(X_sample)
    original_scores = model.predict_risk_score(X_sample)

    save_path = tmp_path / "future_risk_5m.pkl"
    extra_meta = {"horizon": "5m", "horizon_seconds": 300.0}
    model.save(save_path, extra_metadata=extra_meta)

    loaded = RiskModel.load(save_path)
    assert list(loaded.predict(X_sample)) == list(original_preds)
    assert np.allclose(loaded.predict_risk_score(X_sample), original_scores)
    assert loaded.metadata["horizon"] == "5m"
    assert loaded.metadata["horizon_seconds"] == 300.0


# --- MultiHorizonRiskModel ----------------------------------------------------------

def test_multi_horizon_predict_schema(fitted_current_and_5m, small_long_df, feature_columns):
    # Use the same model for "10m" slot too — this test only checks schema shape.
    bundle = MultiHorizonRiskModel(
        current=fitted_current_and_5m["current"],
        future_5m=fitted_current_and_5m["5m"],
        future_10m=fitted_current_and_5m["5m"],
    )
    X, _ = prepare_future_features(small_long_df, "current", feature_columns)
    result = bundle.predict(X.head(3))

    assert set(result.keys()) == {"current", "future_5m", "future_10m"}
    for key in result:
        entry = result[key]
        assert len(entry["risk_class"]) == 3
        assert len(entry["risk_score"]) == 3
        assert len(entry["class_probabilities"]) == 3
        assert len(entry["dangerous_probability"]) == 3
        assert "horizon_seconds" in entry


def test_multi_horizon_idxmax_matches_pipeline_predict(fitted_current_and_5m, small_long_df, feature_columns):
    """Regression test: MultiHorizonRiskModel derives risk_class via
    proba.idxmax() (to avoid a redundant predict_proba() call per model,
    a real inefficiency caught during I1 benchmarking) — must be exactly
    equivalent to the pipeline's own predict()."""
    model = fitted_current_and_5m["current"]
    X, _ = prepare_future_features(small_long_df, "current", feature_columns)
    X_sample = X.head(50)

    proba = model.predict_proba(X_sample)
    via_idxmax = list(proba.idxmax(axis=1))
    via_predict = list(model.predict(X_sample))
    assert via_idxmax == via_predict


def test_dangerous_probability_bounded(fitted_current_and_5m, small_long_df, feature_columns):
    model = fitted_current_and_5m["current"]
    X, _ = prepare_future_features(small_long_df, "current", feature_columns)
    proba = model.predict_proba(X.head(20))
    dp = dangerous_probability(proba)
    assert (dp >= 0.0).all()
    assert (dp <= 1.0 + 1e-9).all()


# --- H. risk score bounds / I. probability validity ---------------------------------

def test_all_horizon_risk_scores_bounded(fitted_current_and_5m, small_long_df, feature_columns):
    for horizon, model in fitted_current_and_5m.items():
        X, _ = prepare_future_features(small_long_df, horizon, feature_columns)
        scores = model.predict_risk_score(X.head(20))
        assert (scores >= 0.0).all()
        assert (scores <= 100.0).all()


def test_all_horizon_probabilities_sum_to_one(fitted_current_and_5m, small_long_df, feature_columns):
    for horizon, model in fitted_current_and_5m.items():
        X, _ = prepare_future_features(small_long_df, horizon, feature_columns)
        proba = model.predict_proba(X.head(20))
        assert np.allclose(proba.sum(axis=1).values, 1.0, atol=1e-6)


# --- J. determinism ------------------------------------------------------------------

def test_same_config_reproducible_predictions(small_long_df, feature_columns):
    X, y = prepare_future_features(small_long_df, "current", feature_columns)
    train_mask = small_long_df.loc[X.index, "split"] == "train"

    def _train():
        pipe = build_hgb_pipeline(15)
        return RiskModel.fit(pipe, X[train_mask], y[train_mask])

    model_a, model_b = _train(), _train()
    assert list(model_a.predict(X.head(20))) == list(model_b.predict(X.head(20)))


# --- horizon metadata sanity ---------------------------------------------------------

def test_horizon_seconds_and_columns_consistent():
    assert HORIZON_SECONDS["current"] == 0.0
    assert HORIZON_SECONDS["5m"] == 300.0
    assert HORIZON_SECONDS["10m"] == 600.0
    assert HORIZON_COLUMNS["current"] == "label"
    assert HORIZON_COLUMNS["5m"] == "future_5min_label"
    assert HORIZON_COLUMNS["10m"] == "future_10min_label"
