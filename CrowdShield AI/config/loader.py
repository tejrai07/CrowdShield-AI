"""Loads and validates YAML config into an AppConfig object.

Supports layering a venue-specific YAML on top of the default (or explicitly
selected) base config, so different deployments don't require duplicating
the whole file. Configuration is loaded exactly once per call — no
duplicate reads.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from config.schema import AppConfig

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge override into base, recursively, without mutating either input."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_path: Path | str | None = None,
    venue_path: Path | str | None = None,
) -> AppConfig:
    """Load configuration exactly once and return a validated AppConfig.

    Behavior:
        no arguments              -> default config
        config_path only          -> that config
        venue_path only           -> default config + venue override
        config_path + venue_path  -> config_path + venue override

    Raises:
        FileNotFoundError if either path doesn't exist.
        pydantic.ValidationError if the merged config fails schema validation.
    """
    base_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw = _read_yaml(base_path)

    if venue_path is not None:
        venue_raw = _read_yaml(Path(venue_path))
        raw = _deep_merge(raw, {"venue": venue_raw})

    return AppConfig(**raw)
