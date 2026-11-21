"""D1 integration test — REAL YOLO detector + REAL ByteTrack tracker +
ZoneManager on the REAL sample video, using the example venue's zones.
Mirrors the opt-in pattern from B1/C1 integration tests.

Enable with:
    CROWDSHIELD_RUN_INTEGRATION=1 pytest tests/test_zones_integration.py
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
EXAMPLE_VENUE = REPO_ROOT / "config" / "venues" / "example_venue.yaml"


def test_real_pipeline_assigns_tracks_to_zones():
    from config.loader import load_config
    from features.zones import ZoneManager
    from vision.benchmark import FrameSource
    from vision.detector import PersonDetector
    from vision.tracker import PersonTracker

    if not TEST_VIDEO.exists():
        pytest.skip(f"Sample video not present at {TEST_VIDEO} — see README for download steps.")

    cfg = load_config(venue_path=EXAMPLE_VENUE)
    detector = PersonDetector(model="yolov8n.pt", confidence=0.4, device="cpu", image_size=640)
    tracker = PersonTracker(max_age_frames=30, trajectory_history_seconds=15.0)
    zone_manager = ZoneManager(cfg.venue.zones)
    source = FrameSource(str(TEST_VIDEO), target_fps=12.0)

    zone_assignments_seen: set[str] = set()
    any_track_seen = False
    frames_checked = 0

    for sampled in source:
        detections = detector.detect(sampled.frame)
        tracks = tracker.update(detections, timestamp_s=sampled.source_timestamp_s)
        for t in tracks:
            any_track_seen = True
            zone_id = zone_manager.assign_track(t)
            # The example venue's 3 zones fully tile the 768x432 frame, so
            # every track center should resolve to a real zone, never None.
            assert zone_id is not None
            zone_assignments_seen.add(zone_id)

        frames_checked += 1
        if frames_checked >= 60:
            break

    assert frames_checked == 60
    assert any_track_seen, "expected at least one track over 60 frames of this clip"
    assert zone_assignments_seen.issubset({"entrance", "main_area", "exit"})
