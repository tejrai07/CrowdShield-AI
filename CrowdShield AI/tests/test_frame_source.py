"""Unit tests for vision/benchmark.py's FrameSource (frame sampling).

Uses a tiny synthetically-generated video (a few solid-color frames written
with OpenCV) rather than the real test_video.mp4, so these tests run fast
and don't depend on the sample dataset being present.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vision.benchmark import FrameSource


@pytest.fixture
def tiny_video(tmp_path: Path) -> str:
    """Write a tiny synthetic video: 30 frames at 30 FPS (1 second), each a
    solid color frame so we can trivially count frames read back."""
    path = tmp_path / "tiny.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (64, 48))
    for i in range(30):
        frame = np.full((48, 64, 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(path)


def test_frame_source_reports_video_properties(tiny_video):
    source = FrameSource(tiny_video, target_fps=30.0)
    assert source.source_fps == pytest.approx(30.0, abs=0.5)
    assert source.width == 64
    assert source.height == 48
    assert source.frame_count == 30


def test_frame_source_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        FrameSource("/nonexistent/path/video.mp4", target_fps=15.0)


def test_frame_source_samples_all_frames_when_target_equals_source(tiny_video):
    source = FrameSource(tiny_video, target_fps=30.0)
    frames = list(source)
    # Allow for +/-1 frame due to encoder rounding.
    assert 28 <= len(frames) <= 30


def test_frame_source_downsamples_when_target_below_source(tiny_video):
    """Source is 30 FPS; sampling at 15 FPS target should yield
    approximately half the frames — this is the concrete example from the
    spec (source 30fps, target 15fps -> ~every second frame)."""
    source = FrameSource(tiny_video, target_fps=15.0)
    frames = list(source)
    assert 12 <= len(frames) <= 16  # ~15, allowing rounding slack


def test_frame_source_does_not_use_sleep(monkeypatch, tiny_video):
    """Sampling must be timestamp/index based, not sleep()-based — verify
    no time.sleep call happens during iteration."""
    import time

    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    source = FrameSource(tiny_video, target_fps=15.0)
    list(source)

    assert sleep_calls == []


def test_frame_source_yields_monotonic_timestamps(tiny_video):
    source = FrameSource(tiny_video, target_fps=30.0)
    timestamps = [f.source_timestamp_s for f in source]
    assert timestamps == sorted(timestamps)


def test_frame_source_is_a_generator_not_eager_list(tiny_video):
    """Memory-safety requirement: iterating must not load the whole video
    into a list internally. We can't directly measure RAM here, but we can
    verify __iter__ returns a generator (lazy), not a pre-built list."""
    import types

    source = FrameSource(tiny_video, target_fps=30.0)
    it = iter(source)
    assert isinstance(it, types.GeneratorType)


def test_frame_source_clamps_target_fps_to_source_fps(tiny_video):
    """Regression test: requesting a target_fps HIGHER than the source
    video's native FPS must not silently imply more unique frames/sec were
    processed than the source could provide. `target_fps` (the configured
    value) is preserved as-is, but `effective_target_fps` — the value
    actually used for sampling — must be clamped to source_fps."""
    source = FrameSource(tiny_video, target_fps=60.0)  # tiny_video is ~30 FPS
    assert source.target_fps == 60.0  # configured value preserved
    assert source.effective_target_fps == pytest.approx(source.source_fps, abs=0.5)
    assert source.effective_target_fps <= source.source_fps + 1e-6


def test_frame_source_no_clamp_when_target_below_source(tiny_video):
    source = FrameSource(tiny_video, target_fps=10.0)  # well below ~30 FPS source
    assert source.effective_target_fps == pytest.approx(10.0, abs=1e-6)

