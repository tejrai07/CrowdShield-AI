"""F1 demo runner: process a video with the full A1-F1 pipeline and print
rolling temporal-window features to the console at a fixed interval.

Console-only, same rationale as scripts/run_metrics_demo.py — F1 output is
numeric, not a new visual layer.

Usage:
    python scripts/run_temporal_demo.py \
        --video sample_data/test_video.mp4 \
        --venue config/venues/example_venue.yaml \
        --print-every 24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.loader import load_config  # noqa: E402
from features.crowd_metrics import CrowdMetricsCalculator  # noqa: E402
from features.temporal import TemporalFeatureBuilder  # noqa: E402
from features.zones import ZoneManager  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from vision.benchmark import FrameSource  # noqa: E402
from vision.detector import PersonDetector  # noqa: E402
from vision.tracker import PersonTracker  # noqa: E402


def format_window(w) -> str:
    dslope = f"{w.density_slope:+.3f}" if w.density_slope is not None else "—"
    sslope = f"{w.speed_slope:+.2f}" if w.speed_slope is not None else "—"
    pressure = f"{w.bottleneck_pressure:.1f}" if w.bottleneck_pressure is not None else "—"
    mature = "mature" if w.is_mature else f"warmup({w.coverage_s:.1f}s)"
    return (
        f"  {w.window_seconds:>2}s [{mature:<14}] n={w.sample_count:<3} "
        f"density_slope={dslope:<7} speed_slope={sslope:<7} pressure={pressure}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CrowdShield F1 temporal features demo (console output)")
    parser.add_argument("--video", default="sample_data/test_video.mp4")
    parser.add_argument("--config", default=None)
    parser.add_argument("--venue", default="config/venues/example_venue.yaml")
    parser.add_argument("--print-every", type=int, default=24, help="print a snapshot every N processed frames")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    logger = setup_logging("INFO")
    venue_path = Path(args.venue) if args.venue else None
    cfg = load_config(args.config, venue_path)

    detector = PersonDetector(
        model=cfg.detection.model, confidence=cfg.detection.confidence,
        device=cfg.detection.device, image_size=cfg.detection.image_size,
    )
    tracker = PersonTracker(
        max_age_frames=cfg.tracking.max_age,
        trajectory_history_seconds=cfg.analytics.trajectory_history_seconds,
    )
    zone_manager = ZoneManager(cfg.venue.zones)
    calculator = CrowdMetricsCalculator(zone_manager)
    builder = TemporalFeatureBuilder(
        zone_ids=[z.id for z in zone_manager.zones],
        windows_seconds=cfg.temporal.windows_seconds,
        density_growth_threshold=cfg.temporal.density_growth_threshold,
        max_history_seconds=cfg.analytics.feature_history_seconds,
    )

    video_path = str(REPO_ROOT / args.video) if not Path(args.video).is_absolute() else args.video
    source = FrameSource(video_path, target_fps=cfg.video.target_fps)
    logger.info("Video: %s | zones=%s | windows=%s", video_path, [z.id for z in zone_manager.zones], cfg.temporal.windows_seconds)

    processed = 0
    for sampled in source:
        if args.max_frames is not None and processed >= args.max_frames:
            break

        detections = detector.detect(sampled.frame)
        tracks = tracker.update(detections, timestamp_s=sampled.source_timestamp_s)
        trajectories = {t.track_id: tracker.get_trajectory(t.track_id) for t in tracks}
        zone_snapshot = calculator.compute_snapshot(tracks, trajectories, timestamp_s=sampled.source_timestamp_s)
        temporal_snapshot = builder.update(zone_snapshot)

        if processed % args.print_every == 0:
            print(f"\n--- t={sampled.source_timestamp_s:.2f}s (frame {processed}) ---")
            for zone_id in sorted(temporal_snapshot):
                features = temporal_snapshot[zone_id]
                print(f"{zone_id} (surge_flag={features.surge_flag}):")
                for window_seconds in sorted(features.windows):
                    print(format_window(features.windows[window_seconds]))

        processed += 1

    logger.info("Processed %d frames.", processed)


if __name__ == "__main__":
    main()
