"""E1 demo runner: process a video with the full A1-E1 pipeline and print
per-zone metric snapshots to the console at a fixed interval.

Deliberately console-only (no video overlay) — E1 metrics are numeric
snapshots, not a new visual layer on top of D1's zone demo, so a simple
printed table is the more honest and lightweight validation tool. See
scripts/run_zone_demo.py for the visual zone-boundary demo.

Usage:
    python scripts/run_metrics_demo.py \
        --video sample_data/test_video.mp4 \
        --venue config/venues/example_venue.yaml \
        --print-every 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.loader import load_config  # noqa: E402
from features.crowd_metrics import CrowdMetricsCalculator  # noqa: E402
from features.zones import ZoneManager  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from vision.benchmark import FrameSource  # noqa: E402
from vision.detector import PersonDetector  # noqa: E402
from vision.tracker import PersonTracker  # noqa: E402


def format_metrics(metrics) -> str:
    speed = f"{metrics.avg_speed_px_s:.1f}px/s" if metrics.avg_speed_px_s is not None else "—"
    direction = f"{metrics.flow_direction_deg:.0f}°" if metrics.flow_direction_deg is not None else "—"
    dvar = f"{metrics.direction_variance:.2f}" if metrics.direction_variance is not None else "—"
    return (
        f"{metrics.zone_id:<12} people={metrics.people_count:<3} "
        f"density={metrics.density:.3f} speed={speed:<10} dir={direction:<6} "
        f"dir_var={dvar:<5} in={metrics.inflow_count} out={metrics.outflow_count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CrowdShield E1 metrics demo (console output)")
    parser.add_argument("--video", default="sample_data/test_video.mp4")
    parser.add_argument("--config", default=None)
    parser.add_argument("--venue", default="config/venues/example_venue.yaml")
    parser.add_argument("--print-every", type=int, default=12, help="print a snapshot every N processed frames")
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

    video_path = str(REPO_ROOT / args.video) if not Path(args.video).is_absolute() else args.video
    source = FrameSource(video_path, target_fps=cfg.video.target_fps)
    logger.info("Video: %s | zones=%s", video_path, [z.id for z in zone_manager.zones])

    processed = 0
    for sampled in source:
        if args.max_frames is not None and processed >= args.max_frames:
            break

        detections = detector.detect(sampled.frame)
        tracks = tracker.update(detections, timestamp_s=sampled.source_timestamp_s)
        trajectories = {t.track_id: tracker.get_trajectory(t.track_id) for t in tracks}
        snapshot = calculator.compute_snapshot(tracks, trajectories, timestamp_s=sampled.source_timestamp_s)

        if processed % args.print_every == 0:
            print(f"\n--- t={sampled.source_timestamp_s:.2f}s (frame {processed}) ---")
            for zone_id in sorted(snapshot):
                print(format_metrics(snapshot[zone_id]))

        processed += 1

    logger.info("Processed %d frames.", processed)


if __name__ == "__main__":
    main()
