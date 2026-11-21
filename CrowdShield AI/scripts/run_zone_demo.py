"""D1 demo runner: process a video with PersonDetector + PersonTracker +
ZoneManager and write an annotated output video showing zone polygons,
track bounding boxes/IDs, and each track's assigned zone.

Thin script, not imported by the pipeline — exists to visually validate
zone assignment. No density/risk overlays (that's Phase E+).

Usage:
    python scripts/run_zone_demo.py \
        --video sample_data/test_video.mp4 \
        --output sample_data/output/zones.mp4 \
        --config config/default.yaml \
        --venue config/venues/example_venue.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config.loader import load_config  # noqa: E402
from features.zones import ZoneManager  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from vision.benchmark import FrameSource  # noqa: E402
from vision.detector import PersonDetector  # noqa: E402
from vision.tracker import PersonTracker  # noqa: E402

# Distinct colors cycled across zones so overlapping/adjacent zones are
# visually distinguishable. BGR (OpenCV convention).
ZONE_COLORS = [
    (255, 160, 0),   # blue-ish
    (0, 200, 255),   # orange
    (0, 220, 0),     # green
    (255, 0, 200),   # magenta
    (0, 255, 255),   # yellow
]


def draw_zones(frame, zone_manager: ZoneManager):
    for i, zone in enumerate(zone_manager.zones):
        if not zone.enabled:
            continue
        color = ZONE_COLORS[i % len(ZONE_COLORS)]
        pts = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        label_pos = (int(zone.polygon[0][0]) + 6, int(zone.polygon[0][1]) + 20)
        cv2.putText(
            frame, zone.name, label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )
    return frame


def draw_tracks_with_zones(frame, tracks, zone_manager: ZoneManager, fps: float):
    for t in tracks:
        p1 = (int(t.x1), int(t.y1))
        p2 = (int(t.x2), int(t.y2))
        zone_id = zone_manager.assign_track(t)

        cv2.rectangle(frame, p1, p2, (0, 200, 0), 2)
        label = f"ID:{t.track_id} zone:{zone_id if zone_id else 'none'}"
        cv2.putText(
            frame, label, (p1[0], max(p1[1] - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA,
        )

    overlay = f"People: {len(tracks)}  FPS: {fps:.1f}"
    cv2.putText(
        frame, overlay, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="CrowdShield D1 zone assignment demo")
    parser.add_argument("--video", default="sample_data/test_video.mp4")
    parser.add_argument("--output", default="sample_data/output/zones.mp4")
    parser.add_argument("--config", default=None)
    parser.add_argument("--venue", default="config/venues/example_venue.yaml")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    logger = setup_logging("INFO")
    venue_path = Path(args.venue) if args.venue else None
    cfg = load_config(args.config, venue_path)

    logger.info("Loading detector: model=%s device=%s", cfg.detection.model, cfg.detection.device)
    detector = PersonDetector(
        model=cfg.detection.model,
        confidence=cfg.detection.confidence,
        device=cfg.detection.device,
        image_size=cfg.detection.image_size,
    )
    logger.info("Resolved device: %s", detector.resolved_device.upper())

    tracker = PersonTracker(
        max_age_frames=cfg.tracking.max_age,
        trajectory_history_seconds=cfg.analytics.trajectory_history_seconds,
    )

    zone_manager = ZoneManager(cfg.venue.zones)
    logger.info(
        "Venue: %s | zones=%s",
        cfg.venue.name, [z.id for z in zone_manager.zones],
    )

    video_path = str(REPO_ROOT / args.video) if not Path(args.video).is_absolute() else args.video
    source = FrameSource(video_path, target_fps=cfg.video.target_fps)
    logger.info(
        "Video: %s | source_fps=%.1f target_fps=%.1f effective_target_fps=%.1f resolution=%dx%d",
        video_path, source.source_fps, cfg.video.target_fps, source.effective_target_fps,
        source.width, source.height,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, source.effective_target_fps, (source.width, source.height)
    )

    processed = 0
    zone_hit_counts: dict[str, int] = {}
    unassigned_count = 0
    t_start = time.perf_counter()
    last_frame_t = t_start

    for sampled in source:
        if args.max_frames is not None and processed >= args.max_frames:
            break

        detections = detector.detect(sampled.frame)
        tracks = tracker.update(detections, timestamp_s=sampled.source_timestamp_s)

        now = time.perf_counter()
        instant_fps = 1.0 / max(now - last_frame_t, 1e-6)
        last_frame_t = now

        frame = draw_zones(sampled.frame, zone_manager)
        frame = draw_tracks_with_zones(frame, tracks, zone_manager, instant_fps)
        writer.write(frame)
        processed += 1

        for t in tracks:
            zone_id = zone_manager.assign_track(t)
            if zone_id is None:
                unassigned_count += 1
            else:
                zone_hit_counts[zone_id] = zone_hit_counts.get(zone_id, 0) + 1

    writer.release()
    total_elapsed = time.perf_counter() - t_start

    logger.info(
        "Processed %d frames in %.2fs (%.2f measured FPS)",
        processed, total_elapsed, processed / total_elapsed if total_elapsed else 0,
    )
    logger.info("Zone hit counts: %s | unassigned: %d", zone_hit_counts, unassigned_count)
    logger.info("Annotated zone video written to: %s", output_path)


if __name__ == "__main__":
    main()
