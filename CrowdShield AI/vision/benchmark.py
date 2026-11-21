"""Video frame streaming + benchmarking utilities.

Two responsibilities live here, deliberately kept together because they're
both "measurement infrastructure" rather than core inference logic:

1. `FrameSource` — streams frames from a video file and samples them
   according to a target processing FPS, WITHOUT loading the whole video
   into memory and WITHOUT using sleep()-based pacing (see module docstring
   in detector.py's design notes / README for rationale).

2. `benchmark_model` — runs a `PersonDetector` over a sampled video and
   reports cold-start vs. warm steady-state timing, split into model
   inference latency vs. end-to-end frame processing latency.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from vision.detector import Detection, PersonDetector

logger = logging.getLogger("crowdshield")


@dataclass(slots=True)
class SampledFrame:
    """A single frame selected for processing, with its source context."""

    frame: np.ndarray
    source_frame_index: int
    source_timestamp_s: float


class FrameSource:
    """Streams frames from a video file, sampling down to a target FPS.

    Sampling is done by comparing each source frame's timestamp against the
    next expected sample time — this is deterministic and independent of
    wall-clock processing speed, unlike a sleep()-based approach. If the
    source video is 30 FPS and target_fps is 15, this yields approximately
    every second frame; if source and target FPS are equal, every frame is
    yielded.

    FPS clamping: `target_fps` cannot exceed the source video's native FPS
    — you cannot sample more unique frames per second than the source
    actually contains. The requested `target_fps` is preserved as-configured
    (`self.target_fps`), but sampling internally uses
    `self.effective_target_fps = min(target_fps, source_fps)`, and a
    WARNING is logged when clamping occurs so a benchmark report never
    silently implies more unique frames/sec were processed than the source
    could provide (e.g. requesting 15 FPS from a 12 FPS source yields at
    most 12 unique frames/sec, not 15).

    Memory-safe: frames are yielded one at a time via a generator. No frame
    buffer accumulates — callers are responsible for keeping only the
    bounded state they need (e.g. the last N seconds for tracking, added in
    Phase C).
    """

    def __init__(self, video_path: str, target_fps: float) -> None:
        self.video_path = video_path
        self.target_fps = target_fps  # configured value, preserved as-is

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        self.source_fps: float = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if self.source_fps <= 0:
            raise ValueError(f"Video reports invalid FPS: {self.source_fps}")

        # Effective sampling rate: never exceeds what the source can provide.
        self.effective_target_fps: float = min(self.target_fps, self.source_fps)
        if self.effective_target_fps < self.target_fps:
            logger.warning(
                "target_fps=%.2f exceeds source_fps=%.2f for %s — clamping "
                "effective processing rate to %.2f FPS.",
                self.target_fps, self.source_fps, video_path, self.effective_target_fps,
            )

    def __iter__(self):
        cap = cv2.VideoCapture(self.video_path)
        try:
            sample_interval_s = 1.0 / self.effective_target_fps
            next_sample_time_s = 0.0
            frame_index = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp_s = frame_index / self.source_fps

                if timestamp_s + 1e-9 >= next_sample_time_s:
                    yield SampledFrame(
                        frame=frame,
                        source_frame_index=frame_index,
                        source_timestamp_s=timestamp_s,
                    )
                    next_sample_time_s += sample_interval_s

                frame_index += 1
        finally:
            cap.release()


@dataclass
class BenchmarkResult:
    model_name: str
    device: str
    image_size: int
    source_fps: float
    target_fps: float
    effective_target_fps: float = 0.0
    processed_frames: int = 0
    cold_start_s: float = 0.0
    warmup_frames: int = 0
    total_inference_time_s: float = 0.0
    total_end_to_end_time_s: float = 0.0
    people_counts: list[int] = field(default_factory=list)

    @property
    def avg_inference_latency_ms(self) -> float:
        if self.processed_frames == 0:
            return 0.0
        return (self.total_inference_time_s / self.processed_frames) * 1000.0

    @property
    def avg_end_to_end_latency_ms(self) -> float:
        if self.processed_frames == 0:
            return 0.0
        return (self.total_end_to_end_time_s / self.processed_frames) * 1000.0

    @property
    def measured_fps(self) -> float:
        """Actual measured processing throughput (frames processed / wall
        time spent processing them) — NOT to be confused with
        `effective_target_fps`, which is the *sampling rate* clamp applied
        by FrameSource when target_fps exceeds the source video's FPS.
        This is typically lower than effective_target_fps, since model
        inference is the bottleneck, not frame availability.
        """
        if self.total_end_to_end_time_s == 0:
            return 0.0
        return self.processed_frames / self.total_end_to_end_time_s

    @property
    def avg_people_detected(self) -> float:
        if not self.people_counts:
            return 0.0
        return sum(self.people_counts) / len(self.people_counts)

    def as_dict(self) -> dict:
        return {
            "model": self.model_name,
            "device": self.device,
            "image_size": self.image_size,
            "source_fps": round(self.source_fps, 2),
            "target_fps": self.target_fps,
            "effective_target_fps": round(self.effective_target_fps, 2),
            "processed_frames": self.processed_frames,
            "cold_start_s": round(self.cold_start_s, 3),
            "avg_inference_latency_ms": round(self.avg_inference_latency_ms, 2),
            "avg_end_to_end_latency_ms": round(self.avg_end_to_end_latency_ms, 2),
            "measured_fps": round(self.measured_fps, 2),
            "avg_people_detected": round(self.avg_people_detected, 2),
        }


def benchmark_model(
    video_path: str,
    model: str,
    confidence: float,
    device: str,
    image_size: int,
    target_fps: float,
    warmup_frames: int = 5,
    max_frames: int | None = None,
) -> BenchmarkResult:
    """Benchmark one model on one video.

    Cold start (model construction + weight load) is measured separately
    and excluded from steady-state inference timing. `warmup_frames` are
    run through the model before measurement begins, so JIT/cache effects
    on the very first inferences don't skew the reported average.
    """
    cold_start_t0 = time.perf_counter()
    detector = PersonDetector(
        model=model, confidence=confidence, device=device, image_size=image_size
    )
    cold_start_s = time.perf_counter() - cold_start_t0

    source = FrameSource(video_path, target_fps=target_fps)

    result = BenchmarkResult(
        model_name=model,
        device=detector.resolved_device,
        image_size=image_size,
        source_fps=source.source_fps,
        target_fps=target_fps,
        effective_target_fps=source.effective_target_fps,
        cold_start_s=cold_start_s,
        warmup_frames=warmup_frames,
    )

    for i, sampled in enumerate(source):
        if i < warmup_frames:
            detector.detect(sampled.frame)  # warm up, not measured
            continue

        if max_frames is not None and result.processed_frames >= max_frames:
            break

        e2e_t0 = time.perf_counter()

        infer_t0 = time.perf_counter()
        detections: list[Detection] = detector.detect(sampled.frame)
        infer_elapsed = time.perf_counter() - infer_t0

        e2e_elapsed = time.perf_counter() - e2e_t0

        result.total_inference_time_s += infer_elapsed
        result.total_end_to_end_time_s += e2e_elapsed
        result.people_counts.append(len(detections))
        result.processed_frames += 1

    return result


def benchmark_detector_vs_tracker(
    video_path: str,
    model: str,
    confidence: float,
    device: str,
    image_size: int,
    target_fps: float,
    max_age_frames: int,
    trajectory_history_seconds: float,
    warmup_frames: int = 5,
    max_frames: int | None = None,
) -> dict:
    """Run the same sampled frames through (a) detector only and (b)
    detector + PersonTracker, to measure ByteTrack's added overhead.

    Two independent passes are used (not a single interleaved pass) so
    that each pipeline's timing isn't affected by the other's state — this
    keeps the comparison clean at the cost of running detection twice.
    """
    # Imported here (not at module top) to avoid vision.benchmark forcing a
    # tracker import for callers that only need plain detection benchmarking.
    from vision.tracker import PersonTracker

    detector_only = benchmark_model(
        video_path=video_path, model=model, confidence=confidence, device=device,
        image_size=image_size, target_fps=target_fps, warmup_frames=warmup_frames,
        max_frames=max_frames,
    )

    cold_start_t0 = time.perf_counter()
    detector = PersonDetector(model=model, confidence=confidence, device=device, image_size=image_size)
    tracker = PersonTracker(
        max_age_frames=max_age_frames, trajectory_history_seconds=trajectory_history_seconds
    )
    cold_start_s = time.perf_counter() - cold_start_t0

    source = FrameSource(video_path, target_fps=target_fps)
    result = BenchmarkResult(
        model_name=model, device=detector.resolved_device, image_size=image_size,
        source_fps=source.source_fps, target_fps=target_fps,
        effective_target_fps=source.effective_target_fps,
        cold_start_s=cold_start_s, warmup_frames=warmup_frames,
    )

    max_simultaneous_tracks = 0
    seen_track_ids: set[int] = set()

    for i, sampled in enumerate(source):
        if i < warmup_frames:
            dets = detector.detect(sampled.frame)
            tracker.update(dets, timestamp_s=sampled.source_timestamp_s)
            continue

        if max_frames is not None and result.processed_frames >= max_frames:
            break

        e2e_t0 = time.perf_counter()
        dets = detector.detect(sampled.frame)
        tracks = tracker.update(dets, timestamp_s=sampled.source_timestamp_s)
        e2e_elapsed = time.perf_counter() - e2e_t0

        result.total_end_to_end_time_s += e2e_elapsed
        result.people_counts.append(len(tracks))
        result.processed_frames += 1

        max_simultaneous_tracks = max(max_simultaneous_tracks, len(tracks))
        seen_track_ids.update(t.track_id for t in tracks)

    return {
        "detector_only": detector_only.as_dict(),
        "detector_plus_tracker": result.as_dict(),
        "tracking_overhead_ms": round(
            result.avg_end_to_end_latency_ms - detector_only.avg_end_to_end_latency_ms, 2
        ),
        "max_simultaneous_tracks": max_simultaneous_tracks,
        "total_unique_track_ids": len(seen_track_ids),
    }


def benchmark_tracker_vs_zones(
    video_path: str,
    model: str,
    confidence: float,
    device: str,
    image_size: int,
    target_fps: float,
    max_age_frames: int,
    trajectory_history_seconds: float,
    zones: list,
    warmup_frames: int = 5,
    max_frames: int | None = None,
) -> dict:
    """Run the same sampled frames through (a) detector + tracker only and
    (b) detector + tracker + zone assignment, to measure D1's added
    overhead on top of C1.

    Two independent passes, same rationale as benchmark_detector_vs_tracker.
    """
    from vision.tracker import PersonTracker

    from features.zones import ZoneManager

    tracker_only = benchmark_detector_vs_tracker(
        video_path=video_path, model=model, confidence=confidence, device=device,
        image_size=image_size, target_fps=target_fps, max_age_frames=max_age_frames,
        trajectory_history_seconds=trajectory_history_seconds, warmup_frames=warmup_frames,
        max_frames=max_frames,
    )["detector_plus_tracker"]

    cold_start_t0 = time.perf_counter()
    detector = PersonDetector(model=model, confidence=confidence, device=device, image_size=image_size)
    tracker = PersonTracker(
        max_age_frames=max_age_frames, trajectory_history_seconds=trajectory_history_seconds
    )
    zone_manager = ZoneManager(zones)
    cold_start_s = time.perf_counter() - cold_start_t0

    source = FrameSource(video_path, target_fps=target_fps)
    result = BenchmarkResult(
        model_name=model, device=detector.resolved_device, image_size=image_size,
        source_fps=source.source_fps, target_fps=target_fps,
        effective_target_fps=source.effective_target_fps,
        cold_start_s=cold_start_s, warmup_frames=warmup_frames,
    )

    zone_hit_counts: dict[str, int] = {}
    unassigned_count = 0

    for i, sampled in enumerate(source):
        if i < warmup_frames:
            dets = detector.detect(sampled.frame)
            tracks = tracker.update(dets, timestamp_s=sampled.source_timestamp_s)
            for t in tracks:
                zone_manager.assign_track(t)
            continue

        if max_frames is not None and result.processed_frames >= max_frames:
            break

        e2e_t0 = time.perf_counter()
        dets = detector.detect(sampled.frame)
        tracks = tracker.update(dets, timestamp_s=sampled.source_timestamp_s)
        for t in tracks:
            zone_id = zone_manager.assign_track(t)
            if zone_id is None:
                unassigned_count += 1
            else:
                zone_hit_counts[zone_id] = zone_hit_counts.get(zone_id, 0) + 1
        e2e_elapsed = time.perf_counter() - e2e_t0

        result.total_end_to_end_time_s += e2e_elapsed
        result.people_counts.append(len(tracks))
        result.processed_frames += 1

    return {
        "detector_plus_tracker": tracker_only,
        "detector_plus_tracker_plus_zones": result.as_dict(),
        "zone_assignment_overhead_ms": round(
            result.avg_end_to_end_latency_ms - tracker_only["avg_end_to_end_latency_ms"], 2
        ),
        "zone_hit_counts": zone_hit_counts,
        "unassigned_track_updates": unassigned_count,
    }
