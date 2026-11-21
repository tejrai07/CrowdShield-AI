"""Loads and validates the G1 synthetic-generation YAML config.

Mirrors `config/loader.py`'s single-read-then-validate pattern.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from data.schemas import GeneratorConfig

DEFAULT_SCENARIOS_PATH = Path(__file__).parent / "scenarios.yaml"


def load_generator_config(path: Path | str = DEFAULT_SCENARIOS_PATH) -> GeneratorConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario config file not found: {path}")

    with path.open("r") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return GeneratorConfig(**raw)
