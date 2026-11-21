"""ML Inference & Replay Worker for CrowdShield AI.

Coordinates video inference via InferencePipeline in a background thread,
or streams pre-computed / simulated L1 frames into the backend state store.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

# Ensure root in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.state import state

logger = logging.getLogger("crowdshield.worker")


class MLWorker:
    """Background worker that continuously processes frames and updates state."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self.status: str = "idle"  # "idle", "running", "stopped", "error"
        self.mode: str = "none"  # "pipeline", "replay"
        self.source: str = ""
        self.frames_processed: int = 0
        self.target_fps: float = 5.0
        self.last_error: str | None = None

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "mode": self.mode,
                "source": self.source,
                "frames_processed": self.frames_processed,
                "target_fps": self.target_fps,
                "last_error": self.last_error,
                "is_running": self._thread is not None and self._thread.is_alive(),
            }

    def stop(self, timeout: float = 3.0) -> bool:
        """Signal worker to stop and wait for completion."""
        self._stop_event.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=timeout)
        with self._lock:
            self.status = "stopped"
            self._thread = None
        return True

    def start_replay(
        self,
        jsonl_path: str | Path,
        loop: bool = True,
        fps: float = 2.0,
    ) -> bool:
        """Start streaming frames from a JSONL file into the backend state."""
        self.stop()
        path = Path(jsonl_path)
        if not path.is_file():
            with self._lock:
                self.status = "error"
                self.last_error = f"JSONL replay file not found: {path}"
            return False

        self._stop_event.clear()
        with self._lock:
            self.status = "running"
            self.mode = "replay"
            self.source = str(path)
            self.frames_processed = 0
            self.target_fps = max(0.5, float(fps))
            self.last_error = None

        self._thread = threading.Thread(
            target=self._run_replay,
            args=(path, loop, self.target_fps),
            daemon=True,
            name="MLWorker-Replay",
        )
        self._thread.start()
        return True

    def _run_replay(self, path: Path, loop: bool, fps: float) -> None:
        delay = 1.0 / fps
        try:
            while not self._stop_event.is_set():
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if self._stop_event.is_set():
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            state.update_frame(record)
                            with self._lock:
                                self.frames_processed += 1
                        except Exception as e:
                            logger.warning("Failed to parse replay line: %s", e)

                        time.sleep(delay)

                if not loop:
                    break
        except Exception as exc:
            with self._lock:
                self.status = "error"
                self.last_error = f"Replay failed: {type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                if self.status != "error":
                    self.status = "stopped"

    def start_video(
        self,
        video_path: str | Path,
        venue_yaml: str | Path | None = None,
        max_frames: int | None = None,
        loop: bool = True,
        fps: float = 5.0,
    ) -> bool:
        """Run full InferencePipeline on a video file in background thread."""
        self.stop()
        vid = Path(video_path)
        if not vid.is_file():
            with self._lock:
                self.status = "error"
                self.last_error = f"Video not found: {vid}"
            return False

        self._stop_event.clear()
        with self._lock:
            self.status = "running"
            self.mode = "pipeline"
            self.source = str(vid)
            self.frames_processed = 0
            self.target_fps = max(1.0, float(fps))
            self.last_error = None

        self._thread = threading.Thread(
            target=self._run_pipeline,
            args=(vid, venue_yaml, max_frames, loop, self.target_fps),
            daemon=True,
            name="MLWorker-Pipeline",
        )
        self._thread.start()
        return True

    def _run_pipeline(
        self,
        video_path: Path,
        venue_yaml: str | Path | None,
        max_frames: int | None,
        loop: bool,
        fps: float,
    ) -> None:
        try:
            from config.loader import load_config
            from inference.pipeline import InferencePipeline
            from vision.benchmark import FrameSource

            venue = Path(venue_yaml) if venue_yaml else _REPO_ROOT / "config" / "venues" / "example_venue.yaml"
            cfg = load_config(venue_path=venue)

            pipeline = InferencePipeline.from_config(cfg, repo_root=_REPO_ROOT)
            delay = 1.0 / fps

            while not self._stop_event.is_set():
                source = FrameSource(str(video_path), target_fps=fps)
                count = 0
                try:
                    for result in pipeline.run(source, max_frames=max_frames):
                        if self._stop_event.is_set():
                            break
                        state.update_frame(result)
                        with self._lock:
                            self.frames_processed += 1
                        count += 1
                        time.sleep(delay)
                finally:
                    close = getattr(source, "close", None)
                    if callable(close):
                        close()

                if not loop or (max_frames is not None and count >= max_frames):
                    break

        except Exception as exc:
            with self._lock:
                self.status = "error"
                self.last_error = f"Pipeline execution failed: {type(exc).__name__}: {exc}"
            logger.error("ML worker pipeline error: %s", exc, exc_info=True)
        finally:
            with self._lock:
                if self.status != "error":
                    self.status = "stopped"


# Global worker instance
worker = MLWorker()
