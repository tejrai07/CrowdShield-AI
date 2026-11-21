"""Risk-specific evaluation metrics for the H1 classifier.

Standard sklearn metrics (accuracy, F1, confusion matrix, log loss) are
used directly from sklearn.metrics where sufficient — this module only
adds the CrowdShield-specific safety-oriented metrics that don't exist in
sklearn: dangerous-class recall/precision, false-safe rate, and ordinal
mean absolute error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from models.features import DANGEROUS_LABELS, LABELS_ORDERED, ORDINAL_MAP


def _as_series(values) -> pd.Series:
    return pd.Series(values).reset_index(drop=True)


def dangerous_recall(y_true, y_pred) -> float:
    """Of all truly HIGH/CRITICAL samples, how many did we detect (predict
    as HIGH or CRITICAL)? The primary safety metric — a model that misses
    dangerous states is unacceptable regardless of overall accuracy."""
    y_true, y_pred = _as_series(y_true), _as_series(y_pred)
    is_dangerous_true = y_true.isin(DANGEROUS_LABELS)
    if is_dangerous_true.sum() == 0:
        return float("nan")
    is_dangerous_pred = y_pred.isin(DANGEROUS_LABELS)
    return float((is_dangerous_true & is_dangerous_pred).sum() / is_dangerous_true.sum())


def dangerous_precision(y_true, y_pred) -> float:
    """Of all samples predicted HIGH/CRITICAL, how many actually were?"""
    y_true, y_pred = _as_series(y_true), _as_series(y_pred)
    is_dangerous_pred = y_pred.isin(DANGEROUS_LABELS)
    if is_dangerous_pred.sum() == 0:
        return float("nan")
    is_dangerous_true = y_true.isin(DANGEROUS_LABELS)
    return float((is_dangerous_true & is_dangerous_pred).sum() / is_dangerous_pred.sum())


def false_safe_rate(y_true, y_pred) -> float:
    """Of all truly HIGH/CRITICAL samples, how many were predicted as
    SAFE or LOW? The most safety-critical failure mode — a genuinely
    dangerous state predicted as safe. Should be as close to 0 as possible."""
    y_true, y_pred = _as_series(y_true), _as_series(y_pred)
    is_dangerous_true = y_true.isin(DANGEROUS_LABELS)
    if is_dangerous_true.sum() == 0:
        return float("nan")
    predicted_safe = y_pred.isin({"SAFE", "LOW"})
    return float((is_dangerous_true & predicted_safe).sum() / is_dangerous_true.sum())


def ordinal_mae(y_true, y_pred) -> float:
    """Mean absolute error on the ordinal class values (SAFE=0..CRITICAL=4).
    Distinguishes a HIGH->MODERATE miss (ordinal distance 1) from the much
    more serious HIGH->SAFE miss (ordinal distance 3)."""
    y_true_ord = _as_series(y_true).map(ORDINAL_MAP)
    y_pred_ord = _as_series(y_pred).map(ORDINAL_MAP)
    return float(np.mean(np.abs(y_true_ord.values - y_pred_ord.values)))


def compute_all_metrics(y_true, y_pred, y_proba: np.ndarray | None = None) -> dict:
    """Full H1 metrics bundle: standard classification metrics + the
    risk-specific ones above. `y_proba` columns must already be in
    LABELS_ORDERED order (see RiskModel.predict_proba)."""
    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS_ORDERED, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=LABELS_ORDERED, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "dangerous_recall": dangerous_recall(y_true, y_pred),
        "dangerous_precision": dangerous_precision(y_true, y_pred),
        "false_safe_rate": false_safe_rate(y_true, y_pred),
        "ordinal_mae": ordinal_mae(y_true, y_pred),
    }

    per_class_precision = precision_score(y_true, y_pred, average=None, labels=LABELS_ORDERED, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=LABELS_ORDERED, zero_division=0)
    metrics["per_class_precision"] = dict(zip(LABELS_ORDERED, [float(v) for v in per_class_precision]))
    metrics["per_class_recall"] = dict(zip(LABELS_ORDERED, [float(v) for v in per_class_recall]))
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=LABELS_ORDERED).tolist()
    metrics["confusion_matrix_labels"] = LABELS_ORDERED

    if y_proba is not None:
        # IMPORTANT: sklearn's log_loss silently mis-scores probabilities
        # when given a non-lexicographic `labels=` order — it emits a
        # UserWarning but does NOT actually reorder columns to match, so
        # passing our ordinal LABELS_ORDERED directly here would produce
        # a wrong (much too high) log loss. Fix: reorder the probability
        # columns into lexicographic order ourselves and pass matching
        # sorted labels — verified against a manual calculation during
        # H1 development.
        sorted_labels = sorted(LABELS_ORDERED)
        col_order = [LABELS_ORDERED.index(label) for label in sorted_labels]
        y_proba_sorted = np.asarray(y_proba)[:, col_order]
        metrics["log_loss"] = float(log_loss(y_true, y_proba_sorted, labels=sorted_labels))

    return metrics


def escalation_recall(y_current_true, y_future_true, y_future_pred) -> float:
    """Phase I1 metric: of all samples where the crowd was SAFE/LOW/
    MODERATE now but genuinely became HIGH/CRITICAL at the target
    horizon, how often did the future model correctly predict that
    escalation (predict HIGH/CRITICAL for that horizon)?

    This is the early-warning-specific complement to `dangerous_recall`
    (which only looks at the future TRUE label, not whether it was a
    genuine escalation from a currently-safe state) — it directly answers
    "does this model give useful advance warning before things go bad",
    which is the core promise of I1's future-risk prediction.

    Returns NaN if there were no genuine escalation cases in the sample
    (nothing to measure recall against) — never a fabricated 0 or 1.
    """
    y_current_true = _as_series(y_current_true)
    y_future_true = _as_series(y_future_true)
    y_future_pred = _as_series(y_future_pred)

    currently_safe = ~y_current_true.isin(DANGEROUS_LABELS)
    genuinely_escalates = currently_safe & y_future_true.isin(DANGEROUS_LABELS)

    if genuinely_escalates.sum() == 0:
        return float("nan")

    correctly_predicted = genuinely_escalates & y_future_pred.isin(DANGEROUS_LABELS)
    return float(correctly_predicted.sum() / genuinely_escalates.sum())
