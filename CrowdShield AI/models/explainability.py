"""Explainability for CrowdShield risk predictions (Phase J1).

Answers: "why did this zone get classified SAFE/LOW/MODERATE/HIGH/
CRITICAL?" and, for I1's future models, "why does the model believe this
zone will become more/less dangerous?" Explanations are grounded in the
ACTUAL features and ACTUAL trained model — never a fabricated
natural-language interpretation.

METHOD CHOICE — investigated, not assumed (see README "Phase J1" for the
full writeup):

    SHAP's TreeExplainer was evaluated first, since it's the standard
    tool for this. It works correctly with our exact pinned
    HistGradientBoostingClassifier (sklearn 1.8.0): verified additivity
    (expected_value + sum(shap_values) reproduces the actual predicted
    class via argmax), verified determinism, ~2.7ms/row. It was NOT
    adopted anyway: `pip install shap` pulled in `numba` + `llvmlite` —
    206MB of an LLVM JIT toolchain — as transitive dependencies, purely
    to explain a small tabular gradient-boosting model. This directly
    conflicts with the project's established lightweight/CPU-friendly
    principle (carried through every phase from B1's model-size
    discipline to I1's "no heavyweight forecasting models"), so it was
    rejected on genuine dependency-weight grounds despite working
    correctly — not because it was hard to use.

    INSTEAD: a dependency-free OCCLUSION (ablation) method. For each
    feature, temporarily replace it with a reference "typical" value
    (the training set's per-feature median — the same "typical value"
    convention already used by H1's median-imputation strategy) and
    re-run predict_proba(). The drop in confidence for the predicted
    class when a feature is reverted to typical is that feature's
    contribution — a feature whose actual value is UNUSUAL (far from
    typical) and whose removal drops confidence a lot is doing real work
    supporting the prediction. This is:
      - grounded directly in the ACTUAL model's ACTUAL behavior (an
        empirical measurement, not an approximation of internal tree
        structure that could drift across sklearn versions),
      - fully deterministic (no sampling, no randomness),
      - fast (all N feature perturbations batched into one predict_proba
        call — ~5.7ms for 76 features on the shipped model),
      - zero new dependencies (uses only RiskModel.predict_proba, already
        present).
    It is a coarser approximation than SHAP's exact game-theoretic
    attribution (it ignores feature interactions beyond the single
    perturbed feature), and this is stated plainly rather than
    overclaiming rigor it doesn't have.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from models.risk_model import RiskModel, risk_score_from_proba

DEFAULT_TOP_K = 5

# Human-readable phrases for the 8 direct E1/F1 metrics (used both
# standalone and as the suffix inside window-prefixed feature names).
_BASE_METRIC_PHRASES: dict[str, str] = {
    "people_count": "people count",
    "density": "crowd density",
    "avg_speed_px_s": "average movement speed",
    "speed_variance": "movement speed variability",
    "direction_variance": "movement direction variance (conflicting movement)",
    "inflow_count": "inflow count",
    "outflow_count": "outflow count",
    "surge_flag": "surge flag",
}

# Human-readable phrases for F1's per-window statistic suffixes.
_WINDOW_SUFFIX_PHRASES: dict[str, str] = {
    "sample_count": "sample count",
    "coverage_s": "window coverage",
    "is_mature": "window maturity",
    "density_mean": "average density",
    "density_max": "peak density",
    "density_min": "minimum density",
    "speed_mean": "average speed",
    "speed_min": "minimum speed",
    "direction_variance_mean": "average direction variance",
    "direction_variance_max": "peak direction variance",
    "density_slope": "density trend",
    "speed_slope": "speed trend",
    "direction_variance_slope": "direction-variance trend",
    "inflow_rate": "inflow rate",
    "outflow_rate": "outflow rate",
    "net_flow_rate": "net flow rate",
    "bottleneck_pressure": "bottleneck pressure",
}

_WINDOW_FEATURE_PATTERN = re.compile(r"^w(\d+)_(.+)$")


def human_readable_feature_name(feature: str) -> str:
    """Deterministic feature-name -> human phrase, built from F1/E1's
    systematic naming pattern (w{N}_{suffix}) rather than a hand-maintained
    per-feature lookup table for all 76 columns — avoids typos/omissions
    and stays correct automatically if F1 adds a new statistic.

    Falls back to the raw feature name (not fabricated text) for any
    feature this project doesn't have a phrase mapping for yet.
    """
    match = _WINDOW_FEATURE_PATTERN.match(feature)
    if match:
        window_seconds, suffix = match.group(1), match.group(2)
        phrase = _WINDOW_SUFFIX_PHRASES.get(suffix)
        if phrase:
            return f"{window_seconds}s {phrase}"
    return _BASE_METRIC_PHRASES.get(feature, feature)


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    feature: str
    human_name: str
    value: float
    reference_value: float
    contribution: float
    """Confidence-in-predicted-class(actual) - confidence(feature reverted
    to reference). Positive = this feature's actual (atypical) value
    supports the prediction; negative = this feature's actual value
    undermines confidence relative to what a typical value would give."""


@dataclass(frozen=True, slots=True)
class RiskExplanation:
    """One prediction's explanation, grounded in the actual model and
    actual feature values — never a fabricated narrative."""

    predicted_class: str
    predicted_probability: float
    risk_score: float
    horizon: str
    horizon_seconds: float
    model_version: str
    top_positive_features: list[FeatureContribution]
    """Up to top_k features whose actual value most supports the prediction."""
    top_negative_features: list[FeatureContribution]
    """Up to top_k features whose actual value most undermines the prediction."""
    feature_contributions: dict[str, float] = field(repr=False)
    """Every feature's contribution, not just the top ones — for
    programmatic consumers that want the full picture."""
    feature_values: dict[str, float] = field(repr=False)
    """The actual observed value of every feature at prediction time."""


class ModelExplainer:
    """Wraps ONE RiskModel with a fixed reference (typical-value) profile,
    built once and reused for every explain() call — never recomputed
    per-prediction (matches the project's "load once, reuse" performance
    principle)."""

    def __init__(
        self,
        risk_model: RiskModel,
        reference_values: pd.Series,
        horizon: str = "current",
        horizon_seconds: float = 0.0,
    ) -> None:
        self.risk_model = risk_model
        self.reference_values = reference_values
        self.horizon = horizon
        self.horizon_seconds = horizon_seconds

    @classmethod
    def from_training_data(
        cls, risk_model: RiskModel, reference_df: pd.DataFrame, horizon: str = "current", horizon_seconds: float = 0.0,
    ) -> "ModelExplainer":
        """Builds the reference (typical-value) profile as the per-feature
        median over `reference_df` (expected to already be restricted to
        the model's own feature columns, e.g. via models.features.prepare_features)."""
        reference_values = reference_df[risk_model.feature_names].median()
        return cls(risk_model, reference_values, horizon=horizon, horizon_seconds=horizon_seconds)

    def explain(self, X_row: pd.DataFrame, top_k: int = DEFAULT_TOP_K) -> RiskExplanation:
        """Explain exactly one row. Raises ValueError for anything else —
        explanations are inherently per-instance, not batched, since the
        top-feature ranking is specific to one prediction.

        Performance note: builds the perturbation matrix with numpy
        (tile + fill_diagonal) rather than a per-feature pandas `.loc`
        loop, and calls predict_proba() exactly once for the actual row
        (deriving both predicted_class and risk_score from that single
        result, same pattern as MultiHorizonRiskModel's I1 fix) — an
        earlier version did both naively and measured ~62ms/call; this
        version measures ~15ms/call for the same 76-feature model, a
        real ~4x improvement caught during J1 benchmarking, not a
        rounding difference.
        """
        if len(X_row) != 1:
            raise ValueError(f"explain() requires exactly 1 row, got {len(X_row)}")

        feature_names = self.risk_model.feature_names
        n_features = len(feature_names)

        X_row = X_row[feature_names].astype(float)
        actual_values = X_row.values[0]  # shape (n_features,)

        actual_proba = self.risk_model.predict_proba(X_row)
        predicted_class = actual_proba.idxmax(axis=1).iloc[0]
        base_confidence = float(actual_proba[predicted_class].iloc[0])
        risk_score = float(risk_score_from_proba(actual_proba)[0])

        # Vectorized perturbation matrix: row i = actual row with feature
        # i reverted to its reference value. np.tile + fill_diagonal is
        # far cheaper than n_features individual pandas .loc assignments.
        reference_arr = self.reference_values[feature_names].values.astype(float)
        perturbed_matrix = np.tile(actual_values, (n_features, 1))
        np.fill_diagonal(perturbed_matrix, reference_arr)
        perturbed = pd.DataFrame(perturbed_matrix, columns=feature_names)

        perturbed_proba = self.risk_model.predict_proba(perturbed)
        perturbed_confidence = perturbed_proba[predicted_class].values

        contributions = base_confidence - perturbed_confidence

        feature_values = {f: float(X_row.iloc[0][f]) for f in feature_names}
        feature_contributions = {f: float(contributions[i]) for i, f in enumerate(feature_names)}

        all_contribs = [
            FeatureContribution(
                feature=f,
                human_name=human_readable_feature_name(f),
                value=feature_values[f],
                reference_value=float(self.reference_values[f]),
                contribution=feature_contributions[f],
            )
            for f in feature_names
        ]

        positive_sorted = sorted((c for c in all_contribs if c.contribution > 0), key=lambda c: -c.contribution)
        negative_sorted = sorted((c for c in all_contribs if c.contribution < 0), key=lambda c: c.contribution)

        return RiskExplanation(
            predicted_class=predicted_class,
            predicted_probability=base_confidence,
            risk_score=risk_score,
            horizon=self.horizon,
            horizon_seconds=self.horizon_seconds,
            model_version=self.risk_model.model_version,
            top_positive_features=positive_sorted[:top_k],
            top_negative_features=negative_sorted[:top_k],
            feature_contributions=feature_contributions,
            feature_values=feature_values,
        )


class MultiHorizonExplainer:
    """Bundles a ModelExplainer per horizon (current/+5m/+10m), built
    once from one shared reference dataset — mirrors
    models.future_risk.MultiHorizonRiskModel's "load once" design so
    explainability doesn't duplicate model-loading/reference-computation
    logic per horizon.
    """

    def __init__(self, current: ModelExplainer, future_5m: ModelExplainer, future_10m: ModelExplainer) -> None:
        self.current = current
        self.future_5m = future_5m
        self.future_10m = future_10m

    @classmethod
    def from_bundle(cls, bundle, reference_df: pd.DataFrame) -> "MultiHorizonExplainer":
        """`bundle` is a models.future_risk.MultiHorizonRiskModel. Not
        type-hinted directly to avoid a circular import
        (models.future_risk already imports models.risk_model)."""
        from models.future_risk import HORIZON_SECONDS

        return cls(
            current=ModelExplainer.from_training_data(bundle.current, reference_df, "current", HORIZON_SECONDS["current"]),
            future_5m=ModelExplainer.from_training_data(bundle.future_5m, reference_df, "5m", HORIZON_SECONDS["5m"]),
            future_10m=ModelExplainer.from_training_data(bundle.future_10m, reference_df, "10m", HORIZON_SECONDS["10m"]),
        )

    def explain_all(self, X_row: pd.DataFrame, top_k: int = DEFAULT_TOP_K) -> dict[str, RiskExplanation]:
        return {
            "current": self.current.explain(X_row, top_k=top_k),
            "future_5m": self.future_5m.explain(X_row, top_k=top_k),
            "future_10m": self.future_10m.explain(X_row, top_k=top_k),
        }


def format_explanation(explanation: RiskExplanation) -> str:
    """Deterministic natural-language formatter. Describes ONLY evidence
    actually present in feature_contributions — never a fabricated
    psychological/narrative claim (e.g. never "people are panicking";
    only precise, measured language like "direction variance elevated
    above typical").
    """
    lines = []
    horizon_label = {
        "current": "Current risk",
        "5m": "Predicted risk in 5 minutes",
        "10m": "Predicted risk in 10 minutes",
    }.get(explanation.horizon, f"Risk (+{explanation.horizon_seconds:.0f}s)")

    lines.append(f"{horizon_label}: {explanation.predicted_class}")
    lines.append(f"Confidence: {explanation.predicted_probability * 100:.0f}%")
    lines.append(f"Risk score: {explanation.risk_score:.1f}/100")
    lines.append("")

    if explanation.top_positive_features:
        lines.append("Primary contributors (supporting this prediction):")
        for i, fc in enumerate(explanation.top_positive_features, start=1):
            lines.append(f"  {i}. {_format_feature_line(fc)}")
    else:
        lines.append("Primary contributors: none identified above the reference baseline.")

    if explanation.top_negative_features:
        lines.append("")
        lines.append("Features working against this prediction:")
        for i, fc in enumerate(explanation.top_negative_features, start=1):
            lines.append(f"  {i}. {_format_feature_line(fc)}")

    return "\n".join(lines)


def _format_feature_line(fc: FeatureContribution) -> str:
    """One feature's line in the human-readable explanation. Handles the
    NaN case explicitly and honestly: a feature can be NaN in the actual
    row (F1's warm-up convention — not enough rolling-window history yet,
    see features/temporal.py) while still having a measurable
    contribution (HistGradientBoostingClassifier routes NaN through a
    learned split direction, so reverting NaN to a reference value can
    genuinely shift the prediction). "Above/below typical" is meaningless
    for a missing value, so this is described as unavailable rather than
    silently printing "nan" or fabricating a comparison."""
    if fc.value != fc.value:  # NaN check without importing math/numpy here
        return f"{fc.human_name}: not available yet (insufficient rolling-window history)"
    direction = "above" if fc.value > fc.reference_value else "below"
    return f"{fc.human_name}: {fc.value:.3f} ({direction} typical {fc.reference_value:.3f})"
