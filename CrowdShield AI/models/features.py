"""Feature selection and preparation for the H1 risk model.

Single source of truth for which G1/F1 columns are legitimate predictive
features vs. metadata/target columns that must NEVER enter the feature
matrix (leakage). Everything downstream — training, evaluation, the
leakage audit test — reads from this module rather than re-deriving the
column list independently.
"""
from __future__ import annotations

import pandas as pd

# Stable ordinal ordering — used for the risk-score calculation and every
# ordinal-distance metric. This exact order (not sklearn's default
# alphabetical string ordering) is what the rest of H1 assumes.
LABELS_ORDERED: list[str] = ["SAFE", "LOW", "MODERATE", "HIGH", "CRITICAL"]
ORDINAL_MAP: dict[str, int] = {label: i for i, label in enumerate(LABELS_ORDERED)}
DANGEROUS_LABELS: set[str] = {"HIGH", "CRITICAL"}

# Columns that must NEVER enter the feature matrix — identifiers,
# metadata, or the label/future-label targets themselves. This set is the
# H1 leakage-prevention mechanism; the automated leakage-audit test
# asserts the trained model's feature_names never intersects this set.
EXCLUDED_COLUMNS: set[str] = {
    "sequence_id", "scenario", "zone_id", "timestamp_s", "data_source",
    "split", "label", "future_5min_label", "future_10min_label",
}

# flow_direction_deg is deliberately excluded from the baseline feature
# set (Option B from the H1 spec, chosen over sin/cos encoding): in this
# single-synthetic-zone setup the absolute direction has no inherent risk
# meaning — it's an arbitrary axis, not oriented to a real venue — and
# direction_variance already captures the risk-relevant "conflicting
# movement" signal on its own. This is a "not predictive/not well-posed"
# exclusion, not a leakage exclusion, so it's tracked separately from
# EXCLUDED_COLUMNS for clarity.
EXCLUDED_NON_PREDICTIVE: set[str] = {"flow_direction_deg"}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns eligible as model features: every dataset column
    except metadata/target columns and the excluded direction column."""
    excluded = EXCLUDED_COLUMNS | EXCLUDED_NON_PREDICTIVE
    return [c for c in df.columns if c not in excluded]


def prepare_features(
    df: pd.DataFrame, feature_columns: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, y_ordinal) ready for model training/evaluation.

    X: predictive features only. Boolean columns (surge_flag, w*_is_mature)
       are cast to int for sklearn compatibility. NaN values from F1's
       warm-up convention (not enough rolling-window history yet) are
       LEFT AS NaN here — imputation, if a model needs it, is that
       model's pipeline's job, not this function's, so different models
       can handle missingness differently (see models/risk_model.py).
    y: the 'label' column (SAFE/LOW/MODERATE/HIGH/CRITICAL strings).
    y_ordinal: y mapped through ORDINAL_MAP — for ordinal-error metrics
       and the manual-scenario sanity checks, not used as a training target.
    """
    if feature_columns is None:
        feature_columns = get_feature_columns(df)

    X = df[feature_columns].copy()
    bool_columns = [c for c in X.columns if X[c].dtype == bool]
    for c in bool_columns:
        X[c] = X[c].astype(int)

    y = df["label"]
    y_ordinal = y.map(ORDINAL_MAP)

    return X, y, y_ordinal
