"""H1 risk model: a thin, serializable wrapper around an sklearn Pipeline.

Not coupled to YOLO/OpenCV/vision in any way — consumes only a flat
feature row/matrix matching F1's TemporalFeatures schema (via
models.features.prepare_features). The eventual K1 real-time pipeline
should be able to do:

    TemporalFeatures -> flat feature row -> RiskModel.predict_risk_score() -> RiskPrediction

IMPORTANT — WHAT THE RISK SCORE IS AND ISN'T:
    predict_risk_score() returns a 0-100 DISPLAY/decision-support score
    derived from class probabilities. It is NOT a probability of injury,
    NOT a probability of a stampede, and NOT a scientifically validated
    physical danger percentage. See README "Phase H1" for the full
    disclaimer.

SERIALIZATION / PORTABILITY (see README "H1 portability fix"):
    save() writes TWO sibling files: a joblib-serialized fitted Pipeline
    (`<path>`) and a plain-JSON metadata sidecar (`<path>.meta.json`)
    containing feature names, model version, class order, and — critically
    — the exact scikit-learn version used to train the model. This split
    exists because a real cross-machine failure was found during
    development: HistGradientBoostingClassifier holds a reference to a
    private internal loss-function object (`sklearn._loss.loss.*`) whose
    module path is not stable across sklearn versions, so a pipeline
    pickled under one sklearn version can fail to unpickle under another
    with a cryptic `ModuleNotFoundError`. The JSON sidecar is always
    readable regardless of sklearn version (no pickle involved), so
    load() can detect a version mismatch and surface it as one clear,
    actionable error message instead of a bare traceback.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from models.features import LABELS_ORDERED, ORDINAL_MAP

MODEL_VERSION = "h1.0.0"

# Ordinal weights for the risk-score formula: SAFE=0/4, LOW=1/4, ...,
# CRITICAL=4/4 — fixed, not learned, not configurable per model instance,
# so every RiskModel produces a score on the same 0-100 scale regardless
# of which underlying classifier is used.
_ORDINAL_WEIGHTS = np.array([ORDINAL_MAP[label] for label in LABELS_ORDERED], dtype=float) / (len(LABELS_ORDERED) - 1)


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def risk_score_from_proba(proba: pd.DataFrame) -> np.ndarray:
    """0-100 DISPLAY score = 100 * sum(P(class_i) * ordinal_i / 4), computed
    directly from an already-computed predict_proba() DataFrame (columns
    in LABELS_ORDERED order). Factored out so callers that need both the
    class prediction and the risk score from one proba call (e.g.
    MultiHorizonRiskModel.predict()) don't pay for predict_proba() twice."""
    return 100.0 * (proba[LABELS_ORDERED].values @ _ORDINAL_WEIGHTS)


class RiskModel:
    """fit() / predict() / predict_proba() / predict_risk_score() /
    evaluate() / save() / load() around a fitted sklearn Pipeline.

    Construct via `RiskModel.fit(...)`, not `RiskModel(...)` directly —
    the constructor assumes an already-fitted pipeline (it reads
    `pipeline.classes_`, which only exists post-fit).
    """

    def __init__(self, pipeline, feature_names: list[str], model_version: str = MODEL_VERSION) -> None:
        self.pipeline = pipeline
        self.feature_names = list(feature_names)
        self.model_version = model_version
        # sklearn orders classes_ alphabetically for string labels (e.g.
        # CRITICAL, HIGH, LOW, MODERATE, SAFE) — NOT our ordinal order.
        # Captured once here so predict_proba()/predict_risk_score() can
        # always reindex into the fixed LABELS_ORDERED order.
        self._sklearn_class_order = list(self.pipeline.classes_)
        self._trained_sklearn_version = sklearn.__version__

    @classmethod
    def fit(cls, pipeline, X_train: pd.DataFrame, y_train: pd.Series, model_version: str = MODEL_VERSION) -> "RiskModel":
        pipeline.fit(X_train, y_train)
        return cls(pipeline, feature_names=list(X_train.columns), model_version=model_version)

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Input is missing required features: {sorted(missing)}")
        return X[self.feature_names]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(self._select_features(X))

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Class probabilities as a DataFrame with columns in the FIXED
        LABELS_ORDERED order (SAFE, LOW, MODERATE, HIGH, CRITICAL) —
        never sklearn's internal alphabetical class order.

        Uses `.reindex(..., fill_value=0.0)` rather than direct column
        indexing: if the training data happened to be missing one class
        entirely (a real edge case caught during J1 testing with a small
        dataset fixture — sklearn's `pipeline.classes_` then only has 4
        classes), this correctly reports 0.0 probability for the unseen
        class instead of raising a KeyError. This is the honest
        representation: the model literally cannot output a probability
        for a class it never saw during training.
        """
        raw = self.pipeline.predict_proba(self._select_features(X))
        raw_df = pd.DataFrame(raw, columns=self._sklearn_class_order, index=X.index)
        return raw_df.reindex(columns=LABELS_ORDERED, fill_value=0.0)

    def predict_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """0-100 DISPLAY score = 100 * sum(P(class_i) * ordinal_i / 4).
        A decision-support number, NOT a physical safety probability —
        see module docstring."""
        return risk_score_from_proba(self.predict_proba(X))

    def evaluate(self, X: pd.DataFrame, y_true) -> dict:
        from evaluation.metrics import compute_all_metrics

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X).values
        return compute_all_metrics(y_true, y_pred, y_proba)

    def save(self, path: Path | str, extra_metadata: dict | None = None) -> None:
        """Writes `<path>` (joblib-serialized Pipeline) and
        `<path>.meta.json` (plain-JSON metadata, including the training-
        time scikit-learn version — see module docstring).

        `extra_metadata` (optional, additive, backward-compatible):
        merged into the JSON sidecar for callers that need to record
        additional provenance — e.g. I1's future-risk models record
        `horizon_seconds`, `training_dataset`, `training_seed`, and
        hyperparameters here rather than H1's serialization mechanism
        being duplicated or changed for that purpose.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pipeline, path)

        metadata = {
            "feature_names": self.feature_names,
            "model_version": self.model_version,
            "sklearn_class_order": self._sklearn_class_order,
            "labels_ordered": LABELS_ORDERED,
            "sklearn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        with _metadata_path(path).open("w") as f:
            json.dump(metadata, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path | str) -> "RiskModel":
        """Loads a RiskModel previously written by save().

        Reads the JSON metadata sidecar FIRST (always succeeds regardless
        of sklearn version, since it's plain JSON, not pickle) and checks
        the recorded training-time scikit-learn version against the
        currently installed one. On a mismatch, this warns immediately,
        and if the subsequent joblib.load() of the pipeline itself then
        fails, the resulting error message names the version mismatch and
        the exact fix (`pip install scikit-learn==<trained_version>`)
        instead of surfacing a bare ModuleNotFoundError.

        TRUST NOTE: this uses `joblib.load` (pickle-based under the
        hood), which can execute arbitrary code for an untrusted file.
        Only ever load artifacts produced by this project's own
        `scripts/train_risk_model.py` — never a model file from an
        untrusted or external source.
        """
        path = Path(path)
        meta_path = _metadata_path(path)
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing metadata sidecar for {path}: expected {meta_path}. "
                f"RiskModel artifacts require both files — re-run scripts/train_risk_model.py."
            )
        with meta_path.open("r") as f:
            metadata = json.load(f)

        installed_version = sklearn.__version__
        trained_version = metadata.get("sklearn_version", "unknown")
        if trained_version != installed_version:
            warnings.warn(
                f"RiskModel at {path} was trained with scikit-learn {trained_version}, "
                f"but the installed version is {installed_version}. scikit-learn does not "
                f"guarantee pickle/joblib compatibility across versions. If loading fails "
                f"below, run: pip install scikit-learn=={trained_version} "
                f"(or regenerate the model with scripts/train_risk_model.py under the "
                f"currently installed version).",
                RuntimeWarning,
                stacklevel=2,
            )

        try:
            pipeline = joblib.load(path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load RiskModel pipeline from {path} "
                f"({type(e).__name__}: {e}). This model was trained with scikit-learn "
                f"{trained_version}; the installed version is {installed_version}. "
                f"scikit-learn model artifacts are only guaranteed compatible within the "
                f"exact same scikit-learn version. Fix: `pip install scikit-learn=={trained_version}` "
                f"(matches requirements.txt's pin), or regenerate the model artifact with "
                f"`python scripts/train_risk_model.py` under the currently installed version."
            ) from e

        instance = cls.__new__(cls)
        instance.pipeline = pipeline
        instance.feature_names = metadata["feature_names"]
        instance.model_version = metadata["model_version"]
        instance._sklearn_class_order = metadata["sklearn_class_order"]
        instance._trained_sklearn_version = trained_version
        instance.metadata = metadata  # full sidecar dict, including any extra_metadata from save()
        return instance

