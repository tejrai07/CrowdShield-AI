"""C1 demo runner: process a video with PersonDetector + PersonTracker and
write an annotated output video (bounding boxes + track ID + confidence +
live people count/FPS overlay).

Thin script, not imported by the pipeline — exists to produce a
human-checkable output for tracking sanity evaluation.

Usage:
    python scripts/run_tracking_demo.py \
        --video sample_data/test_video.mp4 \
        --output sample_data/output/tracked.mp4 \
        --config config/default.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from config.loader import load_config  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from vision.benchmark import FrameSource  # noqa: E402
from vision.detector import PersonDetector  # noqa: E402
from vision.tracker import PersonTracker  # noqa: E402


def draw_tracks(frame, tracks, fps: float):
    for t in tracks:
        p1 = (int(t.x1), int(t.y1))
        p2 = (int(t.x2), int(t.y2))
        cv2.rectangle(frame, p1, p2, (0, 200, 0), 2)
        label = f"ID:{t.track_id} person {t.confidence:.2f}"
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
    parser = argparse.ArgumentParser(description="CrowdShield C1 tracking demo")
    parser.add_argument("--video", default="sample_data/test_video.mp4")
    parser.add_argument("--output", default="sample_data/output/tracked.mp4")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    logger = setup_logging("INFO")
    cfg = load_config(args.config)

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
    logger.info(
        "Tracker: %s | max_age_frames=%d trajectory_history_seconds=%.1f",
        cfg.tracking.tracker, cfg.tracking.max_age, cfg.analytics.trajectory_history_seconds,
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
    max_simultaneous_tracks = 0
    seen_track_ids: set[int] = set()
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

        annotated = draw_tracks(sampled.frame, tracks, instant_fps)
        writer.write(annotated)
        processed += 1

        max_simultaneous_tracks = max(max_simultaneous_tracks, len(tracks))
        seen_track_ids.update(t.track_id for t in tracks)

    writer.release()
    total_elapsed = time.perf_counter() - t_start

    logger.info(
        "Processed %d frames in %.2fs (%.2f measured FPS)",
        processed, total_elapsed, processed / total_elapsed if total_elapsed else 0,
    )
    logger.info(
        "Max simultaneous tracks: %d | Total unique track IDs seen: %d",
        max_simultaneous_tracks, len(seen_track_ids),
    )
    logger.info("Annotated tracking video written to: %s", output_path)


if __name__ == "__main__":
    main()
