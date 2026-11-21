"""H1 manual scenario sanity test (spec section 26).

Trains a small, fast, self-contained model on a small G1 dataset (pure
scenarios only, generated on the fly — deterministic seed, no
transitions, few realizations) and checks RELATIVE risk-score ordering
across scenario types. Deliberately does NOT hardcode exact expected
scores — only sensible relative behavior, since exact values depend on
noise and the specific trained model.
"""
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.loader import load_generator_config
from data.synthetic_generator import generate_dataset
from models.features import prepare_features
from models.risk_model import RiskModel


@pytest.fixture(scope="module")
def scenario_risk_scores() -> pd.DataFrame:
    base_cfg = load_generator_config()
    # Pure scenarios only, several realizations for a stable mean, no
    # transitions (this test cares about steady-state scenario behavior).
    cfg = base_cfg.model_copy(update={
        "realizations_per_scenario": 6, "realizations_per_transition": 0, "duration_s": 60.0,
    })
    df, _ = generate_dataset(cfg)

    X, y, _ = prepare_features(df)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model = RiskModel.fit(pipe, X, y)

    df = df.copy()
    df["risk_score"] = model.predict_risk_score(X)
    return df


def _mean_risk(df: pd.DataFrame, scenario: str) -> float:
    return df[df["scenario"] == scenario]["risk_score"].mean()


def test_normal_stable_has_low_risk(scenario_risk_scores: pd.DataFrame):
    assert _mean_risk(scenario_risk_scores, "NORMAL_STABLE") < 25.0


def test_critical_escalation_has_highest_mean_risk(scenario_risk_scores: pd.DataFrame):
    scenarios = ["NORMAL_STABLE", "GRADUAL_BUILDUP", "RAPID_SURGE", "BOTTLENECK", "CONFLICTING_FLOW", "CRITICAL_ESCALATION"]
    means = {s: _mean_risk(scenario_risk_scores, s) for s in scenarios}
    assert means["CRITICAL_ESCALATION"] == max(means.values())


def test_gradual_buildup_riskier_than_normal_stable(scenario_risk_scores: pd.DataFrame):
    assert _mean_risk(scenario_risk_scores, "GRADUAL_BUILDUP") > _mean_risk(scenario_risk_scores, "NORMAL_STABLE")


def test_rapid_surge_riskier_than_gradual_buildup(scenario_risk_scores: pd.DataFrame):
    assert _mean_risk(scenario_risk_scores, "RAPID_SURGE") > _mean_risk(scenario_risk_scores, "GRADUAL_BUILDUP")


def test_bottleneck_shows_elevated_risk(scenario_risk_scores: pd.DataFrame):
    assert _mean_risk(scenario_risk_scores, "BOTTLENECK") > _mean_risk(scenario_risk_scores, "NORMAL_STABLE")


def test_conflicting_flow_shows_elevated_risk_despite_moderate_density(scenario_risk_scores: pd.DataFrame):
    """The key non-trivial claim: CONFLICTING_FLOW must NOT be treated as
    low-risk just because its density is moderate — direction_variance
    alone should be enough to elevate its risk score above NORMAL_STABLE."""
    assert _mean_risk(scenario_risk_scores, "CONFLICTING_FLOW") > _mean_risk(scenario_risk_scores, "NORMAL_STABLE")


def test_all_scenarios_produce_bounded_risk_scores(scenario_risk_scores: pd.DataFrame):
    assert (scenario_risk_scores["risk_score"] >= 0.0).all()
    assert (scenario_risk_scores["risk_score"] <= 100.0).all()
