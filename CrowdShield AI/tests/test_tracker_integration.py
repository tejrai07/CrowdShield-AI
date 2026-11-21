"""C1 integration test — REAL YOLO detector + REAL ByteTrack tracker on the
REAL sample video. Mirrors test_detector_integration.py's opt-in pattern:
skipped by default, requires network access for model weights on first run.

Enable with:
    CROWDSHIELD_RUN_INTEGRATION=1 pytest tests/test_tracker_integration.py
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


def test_real_detector_plus_tracker_runs_end_to_end():
    from vision.benchmark import FrameSource
    from vision.detector import PersonDetector
    from vision.tracker import PersonTracker

    if not TEST_VIDEO.exists():
        pytest.skip(f"Sample video not present at {TEST_VIDEO} — see README for download steps.")

    detector = PersonDetector(model="yolov8n.pt", confidence=0.4, device="cpu", image_size=640)
    tracker = PersonTracker(max_age_frames=30, trajectory_history_seconds=15.0)
    source = FrameSource(str(TEST_VIDEO), target_fps=12.0)

    all_track_ids: set[int] = set()
    frames_checked = 0
    tracks = []

    for sampled in source:
        detections = detector.detect(sampled.frame)
        tracks = tracker.update(detections, timestamp_s=sampled.source_timestamp_s)
        all_track_ids.update(t.track_id for t in tracks)
        frames_checked += 1
        if frames_checked >= 40:
            break

    assert frames_checked == 40
    assert len(all_track_ids) >= 1, "expected at least one track over 40 frames of this clip"

    for t in tracks:
        assert t.track_id > 0
        assert t.x1 < t.x2 and t.y1 < t.y2
        assert 0.0 <= t.confidence <= 1.0
