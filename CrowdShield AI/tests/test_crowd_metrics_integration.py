"""E1 integration test — REAL YOLO detector + REAL ByteTrack tracker +
ZoneManager + CrowdMetricsCalculator on the REAL sample video. Mirrors the
opt-in pattern from B1/C1/D1 integration tests.

Enable with:
    CROWDSHIELD_RUN_INTEGRATION=1 pytest tests/test_crowd_metrics_integration.py

This test does NOT claim the resulting metrics reflect real-world crowd
conditions — the bundled clip is a sparse pedestrian scene (see B1/C1/D1
known limitations). It only verifies the pipeline runs end-to-end without
crashing and produces well-formed metrics.
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


def test_real_pipeline_produces_well_formed_zone_metrics():
    from config.loader import load_config
    from features.crowd_metrics import CrowdMetricsCalculator
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
    calculator = CrowdMetricsCalculator(zone_manager)
    source = FrameSource(str(TEST_VIDEO), target_fps=12.0)

    any_people_count_positive = False
    any_speed_metric_produced = False
    frames_checked = 0

    for sampled in source:
        detections = detector.detect(sampled.frame)
        tracks = tracker.update(detections, timestamp_s=sampled.source_timestamp_s)
        trajectories = {t.track_id: tracker.get_trajectory(t.track_id) for t in tracks}

        # Must not crash on empty frames (no detections/tracks at all).
        snapshot = calculator.compute_snapshot(tracks, trajectories, timestamp_s=sampled.source_timestamp_s)

        assert set(snapshot.keys()) == {"entrance", "main_area", "exit"}
        for zone_id, metrics in snapshot.items():
            assert metrics.zone_id == zone_id
            assert metrics.people_count >= 0
            assert metrics.density >= 0.0
            if metrics.people_count > 0:
                any_people_count_positive = True
            if metrics.avg_speed_px_s is not None:
                any_speed_metric_produced = True
                assert metrics.avg_speed_px_s >= 0.0
                assert metrics.speed_variance is not None
                assert metrics.speed_variance >= 0.0
            if metrics.flow_direction_deg is not None:
                assert 0.0 <= metrics.flow_direction_deg < 360.0
                assert metrics.direction_variance is not None
                assert 0.0 <= metrics.direction_variance <= 1.0 + 1e-9

        frames_checked += 1
        if frames_checked >= 60:
            break

    assert frames_checked == 60
    assert any_people_count_positive, "expected at least one non-empty zone snapshot over 60 frames"
    assert any_speed_metric_produced, "expected at least one speed measurement once trajectory history accumulates"
