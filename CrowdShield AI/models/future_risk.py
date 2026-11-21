"""Future-risk (5-minute / 10-minute) prediction — Phase I1.

Answers: "given the crowd state NOW, what risk class is likely at
+5 minutes and +10 minutes?" This is a MACHINE-LEARNING FORECASTING
FEATURE trained on G1's synthetic temporal scenarios — NOT a physical or
clinical prediction. See README "Phase I1" for the full disclaimer.

DESIGN — REUSES EXISTING INFRASTRUCTURE, DOES NOT DUPLICATE IT:
    G1 (data/synthetic_generator.py) already generates `future_5min_label`
    and `future_10min_label` columns, computed via a per-sequence
    `pandas.shift()` on the label column — this is already leakage-safe
    by construction: each sequence's DataFrame is built and shifted
    independently (see generate_sequence()), so a future label can NEVER
    be pulled from a different sequence, and rows too close to a
    sequence's end already get NaN (never a fabricated label). I1 adds
    NOTHING to that mechanism — it only:
      1. uses a LONGER `duration_s` when generating the dataset (the
         shipped 120s G1 dataset is shorter than the 300s/600s horizons,
         so it has zero non-null future labels — this is documented in
         G1's own README, not a bug), and
      2. filters out rows where the target horizon's label is NaN before
         training/evaluating (can't train on a missing target).

`models.features.get_feature_columns()` / `prepare_features()` are reused
unchanged — they already exclude `future_5min_label`/`future_10min_label`
(and `label` itself) from the feature matrix, so no new leakage-exclusion
logic was needed here either.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.features import DANGEROUS_LABELS, LABELS_ORDERED, get_feature_columns, prepare_features
from models.risk_model import RiskModel, risk_score_from_proba

# G1's exact column names (data/synthetic_generator.py::FUTURE_HORIZONS_S)
# — reused as-is, not renamed, so there is exactly one source of truth for
# what these columns are called.
HORIZON_COLUMNS: dict[str, str] = {
    "current": "label",
    "5m": "future_5min_label",
    "10m": "future_10min_label",
}
HORIZON_SECONDS: dict[str, float] = {"current": 0.0, "5m": 300.0, "10m": 600.0}


def prepare_future_features(
    df: pd.DataFrame,
    horizon: str,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for one horizon ("current", "5m", or "10m"), with
    rows dropped where the target horizon's label is NaN (i.e. that
    horizon extends past the sequence's end — "horizon truncation", never
    silently interpolated or fabricated).

    X uses the SAME feature columns/exclusion logic as H1
    (models.features.get_feature_columns) regardless of horizon — the
    input features always describe the state AT time t; only the target
    y changes per horizon. This is what guarantees X(t) never contains
    information from t+300s/t+600s: the future columns are excluded from
    X by get_feature_columns() before we ever look at which horizon is
    being trained.
    """
    if horizon not in HORIZON_COLUMNS:
        raise ValueError(f"Unknown horizon {horizon!r}; must be one of {list(HORIZON_COLUMNS)}")

    target_column = HORIZON_COLUMNS[horizon]
    if feature_columns is None:
        feature_columns = get_feature_columns(df)

    X, _, _ = prepare_features(df, feature_columns)
    y = df[target_column]

    valid_mask = y.notna()
    return X[valid_mask], y[valid_mask]


def dangerous_probability(proba: pd.DataFrame) -> pd.Series:
    """Sum of P(HIGH) + P(CRITICAL) from a RiskModel.predict_proba() output."""
    return proba[list(DANGEROUS_LABELS)].sum(axis=1)


class MultiHorizonRiskModel:
    """Bundles the current-state, +5m, and +10m RiskModel instances
    behind one object, so a caller loads models ONCE and reuses them for
    every prediction (never reload per-frame/per-sample — see README
    "Performance").

    Does not duplicate RiskModel's serialization logic — each horizon's
    model is saved/loaded via its own `RiskModel.save()`/`.load()`
    (joblib + JSON metadata sidecar, unchanged from H1's portability
    fix), just under horizon-specific filenames.
    """

    def __init__(self, current: RiskModel, future_5m: RiskModel, future_10m: RiskModel) -> None:
        self.current = current
        self.future_5m = future_5m
        self.future_10m = future_10m

    def predict(self, X: pd.DataFrame) -> dict[str, dict]:
        """Predict all three horizons for the given feature rows (X
        describes the CURRENT state only — the same X is fed to all three
        models; only the target each was trained on differs).

        Calls each model's predict_proba() exactly ONCE and derives both
        risk_class (argmax) and risk_score from that single result —
        NOT via separate predict()/predict_proba()/predict_risk_score()
        calls, which would triple the actual inference cost per model
        (predict_risk_score() itself calls predict_proba() internally).
        This was a real inefficiency caught during I1 benchmarking — see
        README "Phase I1 performance".

        Returns a dict keyed "current"/"future_5m"/"future_10m", each
        with risk_class, risk_score, class_probabilities, and
        dangerous_probability — matching the I1 spec's example schema.
        """
        result: dict[str, dict] = {}
        for key, model, horizon_s in (
            ("current", self.current, HORIZON_SECONDS["current"]),
            ("future_5m", self.future_5m, HORIZON_SECONDS["5m"]),
            ("future_10m", self.future_10m, HORIZON_SECONDS["10m"]),
        ):
            proba = model.predict_proba(X)  # single call — reused for both class and score below
            risk_class = proba.idxmax(axis=1)
            risk_score = risk_score_from_proba(proba)
            result[key] = {
                "risk_class": list(risk_class),
                "risk_score": list(risk_score),
                "class_probabilities": proba.to_dict(orient="records"),
                "dangerous_probability": list(dangerous_probability(proba)),
                "horizon_seconds": horizon_s,
                "model_version": model.model_version,
            }
        return result

    def save(self, output_dir: Path | str, extra_metadata_by_horizon: dict[str, dict] | None = None) -> dict[str, Path]:
        """Saves all three models under the project's H1 naming
        convention: current stays `risk_model.pkl` is NOT touched here
        (H1's own artifact is never overwritten by I1 — see
        scripts/train_future_risk.py, which saves I1's own current-model
        retrain under a distinct filename). This method just saves
        whatever three RiskModel instances it was constructed with, under
        the filenames the caller provides via `paths`.
        """
        output_dir = Path(output_dir)
        extra_metadata_by_horizon = extra_metadata_by_horizon or {}
        paths = {
            "current": output_dir / "future_risk_current.pkl",
            "future_5m": output_dir / "future_risk_5m.pkl",
            "future_10m": output_dir / "future_risk_10m.pkl",
        }
        self.current.save(paths["current"], extra_metadata_by_horizon.get("current"))
        self.future_5m.save(paths["future_5m"], extra_metadata_by_horizon.get("future_5m"))
        self.future_10m.save(paths["future_10m"], extra_metadata_by_horizon.get("future_10m"))
        return paths

    @classmethod
    def load(cls, output_dir: Path | str) -> "MultiHorizonRiskModel":
        output_dir = Path(output_dir)
        return cls(
            current=RiskModel.load(output_dir / "future_risk_current.pkl"),
            future_5m=RiskModel.load(output_dir / "future_risk_5m.pkl"),
            future_10m=RiskModel.load(output_dir / "future_risk_10m.pkl"),
        )
