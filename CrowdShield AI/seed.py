"""Reproducibility helpers.

A1 keeps this standard-library-only: we only need to seed `random` to
validate configuration/logging behavior deterministically. NumPy (and later
torch) seeding will be added here once those libraries actually enter the
active pipeline (Phase B onward) — importing numpy just for A1 would add an
unnecessary dependency to a stage that doesn't use it.
"""
from __future__ import annotations

import random


def set_global_seed(seed: int) -> None:
    random.seed(seed)
