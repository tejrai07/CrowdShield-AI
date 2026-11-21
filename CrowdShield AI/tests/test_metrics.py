"""Unit tests for evaluation/metrics.py.

All synthetic hand-picked label arrays — fast, no model training needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import (
    compute_all_metrics,
    dangerous_precision,
    dangerous_recall,
    false_safe_rate,
    ordinal_mae,
)
from models.features import LABELS_ORDERED


def test_dangerous_recall_all_detected():
    y_true = ["HIGH", "CRITICAL", "HIGH"]
    y_pred = ["HIGH", "CRITICAL", "CRITICAL"]
    assert dangerous_recall(y_true, y_pred) == pytest.approx(1.0)


def test_dangerous_recall_partial_detection():
    y_true = ["HIGH", "CRITICAL", "HIGH", "CRITICAL"]
    y_pred = ["HIGH", "SAFE", "HIGH", "CRITICAL"]  # 3 of 4 correctly flagged dangerous
    assert dangerous_recall(y_true, y_pred) == pytest.approx(0.75)


def test_dangerous_recall_no_dangerous_samples_is_nan():
    y_true = ["SAFE", "LOW", "SAFE"]
    y_pred = ["SAFE", "LOW", "LOW"]
    assert np.isnan(dangerous_recall(y_true, y_pred))


def test_dangerous_precision_correct():
    y_true = ["HIGH", "SAFE", "CRITICAL", "SAFE"]
    y_pred = ["HIGH", "HIGH", "CRITICAL", "SAFE"]  # 2 of 2 dangerous predictions correct... wait 1 false positive
    # predicted dangerous: idx0 (HIGH, correct), idx1 (HIGH, WRONG - true SAFE), idx2 (CRITICAL, correct)
    assert dangerous_precision(y_true, y_pred) == pytest.approx(2 / 3)


def test_dangerous_precision_no_dangerous_predictions_is_nan():
    y_true = ["HIGH", "CRITICAL"]
    y_pred = ["SAFE", "LOW"]
    assert np.isnan(dangerous_precision(y_true, y_pred))


def test_false_safe_rate_zero_when_no_misses():
    y_true = ["HIGH", "CRITICAL", "HIGH"]
    y_pred = ["HIGH", "CRITICAL", "MODERATE"]  # no dangerous predicted as SAFE/LOW
    assert false_safe_rate(y_true, y_pred) == pytest.approx(0.0)


def test_false_safe_rate_detects_dangerous_predicted_safe():
    y_true = ["HIGH", "CRITICAL", "HIGH", "CRITICAL"]
    y_pred = ["SAFE", "CRITICAL", "LOW", "CRITICAL"]  # 2 of 4 dangerous predicted safe/low
    assert false_safe_rate(y_true, y_pred) == pytest.approx(0.5)


def test_false_safe_rate_no_dangerous_samples_is_nan():
    y_true = ["SAFE", "LOW"]
    y_pred = ["SAFE", "SAFE"]
    assert np.isnan(false_safe_rate(y_true, y_pred))


def test_ordinal_mae_perfect_predictions():
    y_true = ["SAFE", "LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert ordinal_mae(y_true, y_true) == pytest.approx(0.0)


def test_ordinal_mae_distinguishes_severity_of_miss():
    # HIGH -> MODERATE (ordinal distance 1) is a smaller error than
    # HIGH -> SAFE (ordinal distance 3).
    mild_miss = ordinal_mae(["HIGH"], ["MODERATE"])
    severe_miss = ordinal_mae(["HIGH"], ["SAFE"])
    assert mild_miss == pytest.approx(1.0)
    assert severe_miss == pytest.approx(3.0)
    assert severe_miss > mild_miss


def test_ordinal_mae_known_value():
    y_true = ["SAFE", "CRITICAL"]
    y_pred = ["LOW", "MODERATE"]
    # |0-1| + |4-2| = 1 + 2 = 3, mean = 1.5
    assert ordinal_mae(y_true, y_pred) == pytest.approx(1.5)


def test_compute_all_metrics_structure_and_bounds():
    y_true = ["SAFE", "LOW", "MODERATE", "HIGH", "CRITICAL"] * 4
    y_pred = ["SAFE", "LOW", "MODERATE", "HIGH", "CRITICAL"] * 4
    y_proba = np.zeros((20, 5))
    for i, label in enumerate(y_true):
        y_proba[i, LABELS_ORDERED.index(label)] = 1.0

    metrics = compute_all_metrics(y_true, y_pred, y_proba)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert 0.0 <= metrics["dangerous_recall"] <= 1.0
    assert metrics["false_safe_rate"] == pytest.approx(0.0)
    assert metrics["ordinal_mae"] == pytest.approx(0.0)
    assert metrics["log_loss"] < 0.01  # near-zero for perfect, confident predictions
    assert set(metrics["per_class_precision"].keys()) == set(LABELS_ORDERED)
    assert len(metrics["confusion_matrix"]) == 5


def test_log_loss_correct_with_ordinal_label_order():
    """Regression test for a real bug caught during H1 development:
    sklearn's log_loss silently mis-scores probabilities when given a
    non-lexicographic `labels=` order (warns but doesn't actually reorder
    columns). This test locks in the fix (reordering columns internally
    before calling log_loss) against a hand-computed expected value."""
    y_true = ["LOW", "SAFE"]
    # Columns in LABELS_ORDERED = [SAFE, LOW, MODERATE, HIGH, CRITICAL]
    y_proba = np.array([
        [0.1, 0.8, 0.05, 0.03, 0.02],  # true=LOW -> P(LOW)=0.8
        [0.7, 0.2, 0.05, 0.03, 0.02],  # true=SAFE -> P(SAFE)=0.7
    ])
    metrics = compute_all_metrics(y_true, ["LOW", "SAFE"], y_proba)
    expected = -(np.log(0.8) + np.log(0.7)) / 2
    assert metrics["log_loss"] == pytest.approx(expected, abs=1e-6)
