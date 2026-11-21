"""Unit tests for vision/tracker.py.

These tests use the REAL ByteTrack implementation (via `PersonTracker`)
but never touch YOLO, video files, or the network — `Detection` objects
are constructed directly by hand. ByteTrack itself is pure Python/NumPy
and fast, so this stays part of the standard fast test suite.

IMPORTANT — ByteTrack new-track confirmation delay: a track only appears
in `update()`'s return value once matched in a SUBSEQUENT frame, except
for detections present in the very first frame ever processed by a given
tracker instance. Tests below are written with this real behavior in
mind — see module docstring in vision/tracker.py for details.
"""
from __future__ import annotations

from vision.detector import Detection
from vision.tracker import PersonTracker, Track


def _det(x1, y1, x2, y2, conf=0.9):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)


def test_same_person_across_consecutive_frames_keeps_same_id():
    tracker = PersonTracker(max_age_frames=30, trajectory_history_seconds=15.0)

    t1 = tracker.update([_det(10, 10, 50, 100)], timestamp_s=0.0)
    t2 = tracker.update([_det(15, 10, 55, 100)], timestamp_s=0.1)
    t3 = tracker.update([_det(20, 10, 60, 100)], timestamp_s=0.2)

    assert len(t1) == 1 and len(t2) == 1 and len(t3) == 1
    assert t1[0].track_id == t2[0].track_id == t3[0].track_id


def test_new_person_gets_a_new_track_id():
    tracker = PersonTracker(max_age_frames=30, trajectory_history_seconds=15.0)

    # Frame 1: only person A. Frame 2: A continues + B appears (unconfirmed
    # yet, per ByteTrack's confirmation rule). Frame 3: B is confirmed.
    tracker.update([_det(10, 10, 50, 100)], timestamp_s=0.0)
    t2 = tracker.update([_det(12, 10, 52, 100), _det(400, 200, 440, 290)], timestamp_s=0.1)
    t3 = tracker.update([_det(14, 10, 54, 100), _det(402, 200, 442, 290)], timestamp_s=0.2)

    ids_frame2 = {t.track_id for t in t2}
    ids_frame3 = {t.track_id for t in t3}

    assert len(ids_frame2) == 1, "new detection should not yet be confirmed on frame 2"
    assert len(ids_frame3) == 2, "new person should be confirmed and tracked by frame 3"
    assert ids_frame2.issubset(ids_frame3), "person A's id must persist into frame 3"


def test_multiple_simultaneous_people_get_distinct_ids():
    tracker = PersonTracker(max_age_frames=30, trajectory_history_seconds=15.0)

    # Frame 1 is special-cased by ByteTrack: detections present in the very
    # first frame activate immediately, so two people apart in frame 1
    # should both appear right away with distinct ids.
    tracks = tracker.update([_det(10, 10, 50, 100), _det(400, 200, 440, 290)], timestamp_s=0.0)

    ids = [t.track_id for t in tracks]
    assert len(tracks) == 2
    assert len(set(ids)) == 2, "two distinct people must not share a track_id"


def test_track_expires_after_max_age_frames():
    tracker = PersonTracker(max_age_frames=3, trajectory_history_seconds=15.0)

    tracker.update([_det(10, 10, 50, 100)], timestamp_s=0.0)
    confirmed = tracker.update([_det(12, 10, 52, 100)], timestamp_s=0.1)
    a_id = confirmed[0].track_id

    # Feed several frames of an unrelated, distant detection so A never
    # matches again — A should stop appearing in the active output well
    # before it's given a chance to falsely re-match.
    still_present = False
    for i in range(6):
        active = tracker.update([_det(500, 300, 540, 390)], timestamp_s=0.2 + i * 0.1)
        if a_id in {t.track_id for t in active}:
            still_present = True

    assert not still_present, "track A must expire from active output once it stops matching"


def test_track_object_fields_are_well_formed():
    tracker = PersonTracker(max_age_frames=30, trajectory_history_seconds=15.0)
    tracks = tracker.update([_det(10, 20, 90, 200, conf=0.77)], timestamp_s=1.5)

    assert len(tracks) == 1
    t = tracks[0]
    assert isinstance(t, Track)
    assert t.x1 < t.x2 and t.y1 < t.y2
    assert 0.0 <= t.confidence <= 1.0
    assert t.age_frames == 1
    assert t.frames_seen == 1
    assert t.last_seen_s == 1.5
    assert t.width == t.x2 - t.x1
    assert t.center == ((t.x1 + t.x2) / 2.0, (t.y1 + t.y2) / 2.0)


def test_trajectory_buffer_stores_timestamps_and_centers():
    tracker = PersonTracker(max_age_frames=30, trajectory_history_seconds=15.0)

    tracker.update([_det(10, 10, 50, 100)], timestamp_s=0.0)
    t2 = tracker.update([_det(20, 10, 60, 100)], timestamp_s=0.5)
    track_id = t2[0].track_id

    trajectory = tracker.get_trajectory(track_id)
    assert len(trajectory) == 2
    for timestamp, cx, cy in trajectory:
        assert isinstance(timestamp, float)
        assert cx > 0 and cy > 0


def test_trajectory_buffer_respects_maximum_duration():
    tracker = PersonTracker(max_age_frames=100, trajectory_history_seconds=1.0)

    # Same stationary detection over 3 seconds, well beyond the 1s window.
    track_id = None
    for i in range(31):
        ts = i * 0.1  # 0.0 .. 3.0s
        tracks = tracker.update([_det(10, 10, 50, 100)], timestamp_s=ts)
        if tracks:
            track_id = tracks[0].track_id

    assert track_id is not None
    trajectory = tracker.get_trajectory(track_id)
    # All remaining points must fall within the configured window of "now".
    latest_ts = trajectory[-1][0]
    for timestamp, _, _ in trajectory:
        assert (latest_ts - timestamp) <= 1.0 + 1e-6


def test_trajectory_entry_removed_after_track_goes_stale():
    tracker = PersonTracker(max_age_frames=100, trajectory_history_seconds=0.5)

    tracker.update([_det(10, 10, 50, 100)], timestamp_s=0.0)
    t2 = tracker.update([_det(12, 10, 52, 100)], timestamp_s=0.1)
    track_id = t2[0].track_id
    assert tracker.get_trajectory(track_id) != []

    # Advance time far past the trajectory window with an unrelated
    # detection so pruning runs but the original track_id is never updated.
    tracker.update([_det(500, 300, 540, 390)], timestamp_s=5.0)

    assert tracker.get_trajectory(track_id) == [], "stale trajectory entries must be pruned"


def test_empty_detections_returns_empty_tracks():
    tracker = PersonTracker(max_age_frames=30, trajectory_history_seconds=15.0)
    tracks = tracker.update([], timestamp_s=0.0)
    assert tracks == []


def test_multiple_tracker_instances_reset_id_counter():
    """Privacy/session-scoping requirement: a new PersonTracker must not
    continue a previous instance's track_id sequence."""
    tracker_a = PersonTracker(max_age_frames=30, trajectory_history_seconds=15.0)
    tracks_a = tracker_a.update([_det(10, 10, 50, 100)], timestamp_s=0.0)

    tracker_b = PersonTracker(max_age_frames=30, trajectory_history_seconds=15.0)
    tracks_b = tracker_b.update([_det(200, 200, 240, 290)], timestamp_s=0.0)

    assert tracks_a[0].track_id == tracks_b[0].track_id == 1, (
        "each PersonTracker instance must start its own id sequence from 1 "
        "(track IDs are session-scoped, not globally persistent identities)"
    )


def test_detection_to_tracker_bbox_conversion_is_correct():
    """Verify Detection (x1,y1,x2,y2) is correctly converted to the
    tracker's internal xywh (center_x, center_y, w, h) representation and
    the resulting Track bbox matches the original detection closely."""
    from vision.tracker import _detections_to_array

    det = _det(x1=20.0, y1=40.0, x2=100.0, y2=240.0, conf=0.81)
    arr = _detections_to_array([det])

    assert arr.xywh.shape == (1, 4)
    cx, cy, w, h = arr.xywh[0]
    assert cx == 60.0  # (20+100)/2
    assert cy == 140.0  # (40+240)/2
    assert w == 80.0  # 100-20
    assert h == 200.0  # 240-40
    assert arr.conf[0] == 0.81 or abs(arr.conf[0] - 0.81) < 1e-5

    # And round-trip through the real tracker on frame 1 (immediate activation)
    tracker = PersonTracker(max_age_frames=30, trajectory_history_seconds=15.0)
    tracks = tracker.update([det], timestamp_s=0.0)
    assert len(tracks) == 1
    t = tracks[0]
    assert abs(t.x1 - 20.0) < 1.0
    assert abs(t.y1 - 40.0) < 1.0
    assert abs(t.x2 - 100.0) < 1.0
    assert abs(t.y2 - 240.0) < 1.0

