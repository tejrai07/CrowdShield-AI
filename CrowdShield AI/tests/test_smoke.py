"""Smoke tests for Phase A.

Deliberately import-light: no ultralytics, no torch, no cv2. These must
run in well under a second so they can gate every future commit cheaply.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from config.loader import load_config
from config.schema import AppConfig
from logging_setup import setup_logging

REPO_ROOT = Path(__file__).parent.parent


def test_config_loads():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)


def test_config_values_accessible():
    cfg = load_config()
    assert cfg.detection.model == "yolov8n.pt"
    assert cfg.tracking.tracker == "bytetrack"
    assert cfg.analytics.trajectory_history_seconds <= 60


def test_venue_override_layers_correctly():
    venue_path = REPO_ROOT / "config" / "venues" / "example_venue.yaml"
    cfg = load_config(venue_path=venue_path)
    assert cfg.venue.name == "example_venue"
    # example_venue.yaml defines 3 demo zones (entrance/main_area/exit) —
    # see config/venues/example_venue.yaml (updated in Phase D1).
    assert len(cfg.venue.zones) == 3
    assert {z.id for z in cfg.venue.zones} == {"entrance", "main_area", "exit"}


def test_explicit_config_plus_venue_override():
    # --config X --venue Y -> X's base values, with venue overridden by Y
    default_path = REPO_ROOT / "config" / "default.yaml"
    venue_path = REPO_ROOT / "config" / "venues" / "example_venue.yaml"
    cfg = load_config(config_path=default_path, venue_path=venue_path)
    assert cfg.detection.model == "yolov8n.pt"  # came from default.yaml
    assert cfg.venue.name == "example_venue"  # came from venue override


def test_required_directories_exist():
    expected_dirs = ["config", "vision", "features", "data", "models", "inference", "evaluation", "tests"]
    for d in expected_dirs:
        assert (REPO_ROOT / d).is_dir(), f"Missing directory: {d}"


def test_logging_initializes():
    logger = setup_logging("DEBUG")
    assert logger.name == "crowdshield"
    assert logger.level == 10  # DEBUG


def test_config_module_has_no_heavy_imports():
    """Guard rail: importing config.loader/config.schema must never pull in
    ultralytics/torch/cv2/numpy/pandas.

    Run in an isolated subprocess (rather than checking sys.modules in the
    current pytest process) so this test stays robust as the suite grows —
    it validates ONLY what config modules themselves import, not whatever
    else pytest or other test modules happened to load first.
    """
    heavy_modules = ["torch", "ultralytics", "cv2", "numpy", "pandas"]
    check_script = (
        "import sys; "
        "sys.path.insert(0, %r); "
        "import config.loader, config.schema; "
        "loaded = [m for m in %r if m in sys.modules]; "
        "print(','.join(loaded))"
    ) % (str(REPO_ROOT), heavy_modules)

    result = subprocess.run(
        [sys.executable, "-c", check_script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=10,
    )
    assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
    loaded = result.stdout.strip()
    assert loaded == "", f"config modules triggered heavy imports: {loaded}"
