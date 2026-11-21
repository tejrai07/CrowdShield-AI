"""Multi-object tracking abstraction over Ultralytics' ByteTrack.

Like `vision/detector.py`, this module is the ONLY place that touches
ultralytics' tracking internals. Downstream code (Phase D zones, Phase E
crowd metrics) consumes only the lightweight `Track` objects defined here.

Tracker choice: ByteTrack (not BoT-SORT/DeepSORT). We only need short-term
trajectory continuity for speed/direction/flow calculations — not identity
persistence across long occlusions or camera cuts — so ByteTrack's
motion-only (IoU) association is sufficient and avoids the cost of an
appearance re-identification model.

PRIVACY: track_id values are purely technical, session-scoped integers
assigned by ByteTrack's internal counter. They:
  - exist only for the lifetime of a single PersonTracker instance
    (the ID counter resets to 0 whenever a new PersonTracker is created —
    see `BaseTrack.reset_id()` called during construction);
  - are never derived from or linked to face/biometric/appearance data
    (ByteTrack here uses only bounding-box motion/IoU, no embeddings);
  - are never persisted to disk as identity information by this module;
  - carry no meaning beyond "this bounding box was probably the same
    physical person as that bounding box a moment ago."

TRACKING QUALITY (expected, not promised): same person across consecutive
frames should retain the same ID; a short occlusion (a few frames) may be
bridged via the lost-track buffer (`max_age_frames`); heavy or prolonged
occlusion, crossing paths in a dense crowd, or a person leaving and
re-entering the frame will likely produce a NEW track_id. This is a known
and accepted limitation of motion-only tracking — perfect identity
persistence would require re-identification, which we deliberately do not
use (see module docstring above).

NEW TRACK CONFIRMATION DELAY: this is standard ByteTrack behavior, not a
bug — a brand-new track only appears in `update()`'s returned list once it
has been matched again in a SUBSEQUENT frame (i.e. a detection must be
seen at least twice before it's reported as an active track), except for
detections present in the very first frame the tracker ever processes,
which activate immediately. Concretely: if a new person enters on frame N,
`update()` will not include them until frame N+1 confirms the match. This
avoids creating spurious tracks from one-off false-positive detections.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from vision.detector import Detection


@dataclass(frozen=True, slots=True)
class Track:
    """A single tracked person in the current frame.

    Bounding box convention matches `Detection` in detector.py: pixel
    coordinates, (x1, y1) top-left / (x2, y2) bottom-right, origin at the
    frame's top-left.
    """

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    age_frames: int
    """Frames elapsed since this track_id was first created."""
    frames_seen: int
    """Number of frames in which this track_id has been matched/active."""
    last_seen_s: float
    """Timestamp (source video seconds) of this update."""

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class _DetectionArray:
    """Minimal array-like adapter exposing exactly the interface
    Ultralytics' BYTETracker expects from a "results" object: `.xywh`,
    `.conf`, `.cls`, `len()`, and boolean-mask indexing.

    This keeps our plain `Detection` list decoupled from ultralytics'
    Results/Boxes types — the adapter is constructed fresh from our own
    data on every update() call and discarded immediately after.
    """

    __slots__ = ("xywh", "conf", "cls")

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: np.ndarray) -> "_DetectionArray":
        return _DetectionArray(self.xywh[mask], self.conf[mask], self.cls[mask])


def _detections_to_array(detections: list[Detection]) -> _DetectionArray:
    if not detections:
        return _DetectionArray(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    xywh = np.array(
        [[d.x1 + d.width / 2.0, d.y1 + d.height / 2.0, d.width, d.height] for d in detections],
        dtype=np.float32,
    )
    conf = np.array([d.confidence for d in detections], dtype=np.float32)
    cls = np.array([float(d.class_id) for d in detections], dtype=np.float32)
    return _DetectionArray(xywh, conf, cls)


class PersonTracker:
    """Assigns short-lived, consistent track IDs to person detections
    across frames using ByteTrack.

    Usage:
        tracker = PersonTracker(max_age_frames=30, trajectory_history_seconds=15)
        tracks = tracker.update(detections, timestamp_s=1.23)
        trajectory = tracker.get_trajectory(tracks[0].track_id)
    """

    def __init__(
        self,
        max_age_frames: int = 30,
        trajectory_history_seconds: float = 15.0,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.25,
        match_thresh: float = 0.8,
        fuse_score: bool = True,
    ) -> None:
        # Imported lazily so `import vision.tracker` (e.g. for the `Track`
        # dataclass) doesn't force an ultralytics import for callers that
        # only need the type.
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.max_age_frames = max_age_frames
        self.trajectory_history_seconds = trajectory_history_seconds

        args = SimpleNamespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=max_age_frames,
            match_thresh=match_thresh,
            fuse_score=fuse_score,
        )
        self._tracker = BYTETracker(args)  # resets the global track_id counter to 0

        self._frame_index = 0
        # Bounded per track_id state — pruned every update() so memory
        # never grows unboundedly across a long-running session.
        self._trajectories: dict[int, deque[tuple[float, float, float]]] = {}
        self._first_seen_frame: dict[int, int] = {}
        self._frames_seen: dict[int, int] = {}

    def update(self, detections: list[Detection], timestamp_s: float) -> list[Track]:
        """Advance the tracker by one frame and return currently active tracks."""
        self._frame_index += 1

        det_array = _detections_to_array(detections)
        raw_tracks = self._tracker.update(det_array)  # (N, 8): x1,y1,x2,y2,id,score,cls,idx

        tracks: list[Track] = []
        for row in raw_tracks:
            x1, y1, x2, y2, track_id_f, score, _cls, _idx = row
            track_id = int(track_id_f)

            if track_id not in self._first_seen_frame:
                self._first_seen_frame[track_id] = self._frame_index
                self._frames_seen[track_id] = 0
            self._frames_seen[track_id] += 1

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            self._trajectories.setdefault(track_id, deque()).append((timestamp_s, cx, cy))

            tracks.append(
                Track(
                    track_id=track_id,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(score),
                    age_frames=self._frame_index - self._first_seen_frame[track_id] + 1,
                    frames_seen=self._frames_seen[track_id],
                    last_seen_s=timestamp_s,
                )
            )

        self._prune_trajectories(timestamp_s)
        return tracks

    def get_trajectory(self, track_id: int) -> list[tuple[float, float, float]]:
        """Return the bounded (timestamp, center_x, center_y) history for a
        track_id, oldest first. Pixel-space coordinates — NOT meters; venue
        calibration happens in a later phase.
        """
        return list(self._trajectories.get(track_id, ()))

    def _prune_trajectories(self, now_s: float) -> None:
        """Drop trajectory points (and eventually whole entries) older than
        `trajectory_history_seconds`, for every track_id we know about —
        active or lost. This bounds memory regardless of how long the
        session runs or how many distinct people have been seen, without
        depending on ByteTrack ever telling us a track was formally
        removed.
        """
        stale_ids = []
        for track_id, points in self._trajectories.items():
            while points and (now_s - points[0][0]) > self.trajectory_history_seconds:
                points.popleft()
            if not points:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            del self._trajectories[track_id]
            self._first_seen_frame.pop(track_id, None)
            self._frames_seen.pop(track_id, None)
