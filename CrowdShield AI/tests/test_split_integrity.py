"""Split integrity and leakage-audit tests for H1.

Uses the ACTUAL packaged G1 dataset (sample_data/synthetic/synthetic_dataset.csv)
— these tests re-verify G1's own guarantee rather than trusting it
blindly, per the H1 spec's mandatory leakage-audit requirement.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.synthetic_generator import load_dataset
from models.features import EXCLUDED_COLUMNS, get_feature_columns

REPO_ROOT = Path(__file__).parent.parent
DATASET_PATH = REPO_ROOT / "sample_data" / "synthetic" / "synthetic_dataset.csv"


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        pytest.skip(f"Packaged dataset not present at {DATASET_PATH} — run scripts/generate_synthetic_data.py first.")
    return load_dataset(DATASET_PATH)


def test_train_val_test_sequence_ids_fully_disjoint(dataset: pd.DataFrame):
    seq_by_split = {
        split: set(dataset[dataset["split"] == split]["sequence_id"])
        for split in ["train", "val", "test"]
    }
    assert seq_by_split["train"].isdisjoint(seq_by_split["val"])
    assert seq_by_split["train"].isdisjoint(seq_by_split["test"])
    assert seq_by_split["val"].isdisjoint(seq_by_split["test"])


def test_every_row_has_a_split_assignment(dataset: pd.DataFrame):
    assert dataset["split"].notna().all()
    assert set(dataset["split"].unique()).issubset({"train", "val", "test"})


def test_feature_columns_never_include_excluded_metadata(dataset: pd.DataFrame):
    feature_columns = get_feature_columns(dataset)
    leaked = EXCLUDED_COLUMNS & set(feature_columns)
    assert leaked == set(), f"LEAKAGE: {leaked} found in feature columns"


def test_feature_columns_never_include_future_labels(dataset: pd.DataFrame):
    feature_columns = get_feature_columns(dataset)
    assert "future_5min_label" not in feature_columns
    assert "future_10min_label" not in feature_columns


def test_label_column_not_duplicated_as_a_feature(dataset: pd.DataFrame):
    feature_columns = get_feature_columns(dataset)
    assert "label" not in feature_columns


def test_data_source_column_confirms_synthetic(dataset: pd.DataFrame):
    assert (dataset["data_source"] == "synthetic").all()
