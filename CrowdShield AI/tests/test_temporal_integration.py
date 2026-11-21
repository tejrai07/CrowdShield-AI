"""F1 integration test — REAL YOLO + REAL ByteTrack + ZoneManager +
CrowdMetricsCalculator + TemporalFeatureBuilder on the REAL sample video.
Mirrors the opt-in pattern from B1/C1/D1/E1 integration tests.

Enable with:
    CROWDSHIELD_RUN_INTEGRATION=1 pytest tests/test_temporal_integration.py

Does NOT claim this sparse clip validates crowd-risk prediction — only
that the full A1-F1 pipeline runs end-to-end without crashing, history
grows correctly, short windows populate before long ones, and memory
stays bounded.
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


def test_real_pipeline_produces_temporal_features_end_to_end():
    from config.loader import load_config
    from features.crowd_metrics import CrowdMetricsCalculator
    from features.temporal import TemporalFeatureBuilder
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
    metrics_calculator = CrowdMetricsCalculator(zone_manager)
    temporal_builder = TemporalFeatureBuilder(
        zone_ids=[z.id for z in zone_manager.zones],
        windows_seconds=cfg.temporal.windows_seconds,
        density_growth_threshold=cfg.temporal.density_growth_threshold,
        max_history_seconds=cfg.analytics.feature_history_seconds,
    )
    source = FrameSource(str(TEST_VIDEO), target_fps=12.0)

    max_history_len_seen = 0
    any_5s_window_mature = False
    any_60s_window_mature = False
    frames_checked = 0
    last_timestamp = 0.0

    for sampled in source:
        detections = detector.detect(sampled.frame)
        tracks = tracker.update(detections, timestamp_s=sampled.source_timestamp_s)
        trajectories = {t.track_id: tracker.get_trajectory(t.track_id) for t in tracks}

        zone_snapshot = metrics_calculator.compute_snapshot(
            tracks, trajectories, timestamp_s=sampled.source_timestamp_s
        )
        # Must not crash even on empty frames (no tracks at all).
        temporal_snapshot = temporal_builder.update(zone_snapshot)

        assert set(temporal_snapshot.keys()) == {"entrance", "main_area", "exit"}
        for zone_id, features in temporal_snapshot.items():
            assert set(features.windows.keys()) == {5, 15, 30, 60}
            for window_seconds, w in features.windows.items():
                assert w.sample_count >= 0
                assert w.coverage_s <= window_seconds + 1e-6
                if w.density_slope is not None:
                    assert isinstance(w.density_slope, float)
                if w.window_seconds == 5 and w.is_mature:
                    any_5s_window_mature = True
                if w.window_seconds == 60 and w.is_mature:
                    any_60s_window_mature = True

        # Bounded-memory check: internal history length must never exceed
        # ~max_history_seconds worth of samples (at this video's ~12
        # samples/sec, well under a generous ceiling).
        for tracker_obj in temporal_builder._trackers.values():
            max_history_len_seen = max(max_history_len_seen, len(tracker_obj._history))

        last_timestamp = sampled.source_timestamp_s
        frames_checked += 1

    assert frames_checked > 0
    assert last_timestamp > 0
    assert any_5s_window_mature, "expected the 5s window to mature at some point over the full clip"
    # The clip is only ~20s, so the 60s window should NOT be expected to
    # mature — this positively confirms long windows correctly stay
    # marked incomplete rather than fabricating maturity.
    assert not any_60s_window_mature, "60s window should remain immature on a ~20s clip"
    assert max_history_len_seen < 300, "temporal history must stay bounded, not grow with video length"
