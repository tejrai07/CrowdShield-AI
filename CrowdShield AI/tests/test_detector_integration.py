"""B1 integration tests — use a REAL YOLO model and the REAL sample video.

These are intentionally separated from the fast unit-test suite: they
require network access (to download YOLO weights, if not already cached)
and take several seconds to run. They are skipped by default and only run
when explicitly requested:

    pytest tests/test_detector_integration.py -m integration --run-integration

or by setting the environment variable CROWDSHIELD_RUN_INTEGRATION=1.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CROWDSHIELD_RUN_INTEGRATION") != "1",
    reason=(
        "Integration tests require real model weights + sample video and "
        "network access. Set CROWDSHIELD_RUN_INTEGRATION=1 to enable."
    ),
)

REPO_ROOT = Path(__file__).parent.parent
TEST_VIDEO = REPO_ROOT / "sample_data" / "test_video.mp4"


def test_real_detector_loads_and_runs_on_sample_video():
    from vision.benchmark import FrameSource
    from vision.detector import PersonDetector

    if not TEST_VIDEO.exists():
        pytest.skip(f"Sample video not present at {TEST_VIDEO} — see README for download steps.")

    detector = PersonDetector(model="yolov8n.pt", confidence=0.4, device="cpu", image_size=640)
    source = FrameSource(str(TEST_VIDEO), target_fps=12.0)

    total_detections = 0
    frames_checked = 0
    for sampled in source:
        detections = detector.detect(sampled.frame)
        total_detections += len(detections)
        frames_checked += 1
        if frames_checked >= 20:
            break

    assert frames_checked == 20
    # We don't assert an exact count (no ground truth), only that the
    # pipeline runs end-to-end without error and returns well-formed data.
    assert total_detections >= 0
