"""Tracker tests (FR-TRK-01, FR-TRK-02, FR-TRK-04, FR-TRK-05).

Synthetic detection sequences only — no camera, no model (NFR-TST-01).

The behaviour these protect is the Phase 1 exit criterion: one physical
pothole holds one track ID across its visible frames. Every duplicate ID is a
duplicate alert and a duplicate report.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from roadwatch.core.contracts import BBox, Detection, HazardClass, TrackState
from roadwatch.tracking.tracker import Tracker, iou

T0 = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)


def _det(
    frame_id: int,
    box: tuple[float, float, float, float],
    *,
    conf: float = 0.9,
    hazard: HazardClass = HazardClass.POTHOLE,
) -> Detection:
    x1, y1, x2, y2 = box
    return Detection(
        frame_id=frame_id,
        timestamp=T0 + timedelta(seconds=frame_id / 30),
        hazard_class=hazard,
        confidence=conf,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        model_version="test-v1",
    )


def _ts(frame_id: int) -> datetime:
    return T0 + timedelta(seconds=frame_id / 30)


# --------------------------------------------------------------------------
# IoU
# --------------------------------------------------------------------------


def test_iou_identical_boxes() -> None:
    b = BBox(x1=0, y1=0, x2=10, y2=10)
    assert iou(b, b) == pytest.approx(1.0)


def test_iou_disjoint_boxes() -> None:
    assert iou(BBox(x1=0, y1=0, x2=10, y2=10), BBox(x1=50, y1=50, x2=60, y2=60)) == 0.0


def test_iou_touching_edges_is_zero() -> None:
    assert iou(BBox(x1=0, y1=0, x2=10, y2=10), BBox(x1=10, y1=0, x2=20, y2=10)) == 0.0


def test_iou_half_overlap() -> None:
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    b = BBox(x1=5, y1=0, x2=15, y2=10)
    # intersection 50, union 150
    assert iou(a, b) == pytest.approx(50 / 150)


# --------------------------------------------------------------------------
# Identity — FR-TRK-01
# --------------------------------------------------------------------------


def test_one_pothole_holds_one_track_id() -> None:
    """The Phase 1 exit criterion, as a unit test.

    Simulates a pothole approaching: the box moves down the frame and grows,
    as it does on a real forward-facing camera.
    """
    tracker = Tracker(min_hits=3)
    ids: set[str] = set()

    for i in range(20):
        y = 200 + i * 12
        size = 40 + i * 3
        tracker.update([_det(i, (300, y, 300 + size, y + size))], _ts(i))
        assert len(tracker.active) == 1
        ids.add(tracker.active[0].track_id)

    assert len(ids) == 1, f"expected one stable track id, got {len(ids)}"


def test_two_separate_potholes_get_separate_ids() -> None:
    tracker = Tracker(min_hits=1)
    for i in range(5):
        tracker.update(
            [
                _det(i, (100, 200, 150, 250)),
                _det(i, (500, 200, 550, 250)),
            ],
            _ts(i),
        )
    assert len({t.track_id for t in tracker.active}) == 2


def test_different_hazard_classes_never_merge() -> None:
    """A crack and a pothole in the same place are two different defects."""
    tracker = Tracker(min_hits=1)
    tracker.update([_det(0, (100, 100, 200, 200), hazard=HazardClass.POTHOLE)], _ts(0))
    tracker.update([_det(1, (100, 100, 200, 200), hazard=HazardClass.CRACK)], _ts(1))
    assert len(tracker.active) == 2


def test_track_survives_a_missed_frame() -> None:
    """Kalman prediction bridges a dropped detection without a new ID."""
    tracker = Tracker(min_hits=2, max_age=5)

    for i in range(4):
        tracker.update([_det(i, (300, 200 + i * 10, 360, 260 + i * 10))], _ts(i))
    original = tracker.active[0].track_id

    tracker.update([], _ts(4))  # detector missed this frame

    tracker.update([_det(5, (300, 250, 360, 310))], _ts(5))
    assert len(tracker.active) == 1
    assert tracker.active[0].track_id == original


# --------------------------------------------------------------------------
# Confirmation — FR-TRK-02
# --------------------------------------------------------------------------


def test_single_frame_detection_is_not_confirmed() -> None:
    """Suppresses single-frame false positives, which are common on shadows."""
    tracker = Tracker(min_hits=3)
    tracker.update([_det(0, (100, 100, 150, 150))], _ts(0))
    assert tracker.active[0].state is TrackState.UNKNOWN
    assert tracker.confirmed == []


def test_track_promoted_after_min_hits() -> None:
    tracker = Tracker(min_hits=3)
    for i in range(3):
        tracker.update([_det(i, (100, 100 + i, 150, 150 + i))], _ts(i))
    assert tracker.active[0].state is TrackState.DETECTED
    assert len(tracker.confirmed) == 1


def test_min_hits_is_configurable() -> None:
    """No hard-coded thresholds (non-negotiable rule 5)."""
    tracker = Tracker(min_hits=5)
    for i in range(4):
        tracker.update([_det(i, (100, 100 + i, 150, 150 + i))], _ts(i))
    assert tracker.confirmed == []
    tracker.update([_det(4, (100, 104, 150, 154))], _ts(4))
    assert len(tracker.confirmed) == 1


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        Tracker(iou_threshold=0.0)
    with pytest.raises(ValueError, match="min_hits"):
        Tracker(min_hits=0)


# --------------------------------------------------------------------------
# Loss — FR-TRK-04
# --------------------------------------------------------------------------


def test_track_goes_lost_after_max_age() -> None:
    tracker = Tracker(min_hits=1, max_age=3)
    tracker.update([_det(0, (100, 100, 150, 150))], _ts(0))
    assert len(tracker.active) == 1

    for i in range(1, 6):
        tracker.update([], _ts(i))

    assert tracker.active == []


def test_lost_track_that_captured_still_owes_a_report() -> None:
    """FR-TRK-04: leaving the frame does not cancel a pending report."""
    tracker = Tracker(min_hits=1, max_age=2)
    tracker.update([_det(0, (100, 100, 150, 150))], _ts(0))
    tracker.active[0].has_been_captured = True

    for i in range(1, 5):
        tracker.update([], _ts(i))

    pending = tracker.take_lost()
    assert len(pending) == 1
    assert pending[0].has_been_captured


def test_lost_track_without_capture_owes_nothing() -> None:
    tracker = Tracker(min_hits=1, max_age=2)
    tracker.update([_det(0, (100, 100, 150, 150))], _ts(0))
    for i in range(1, 5):
        tracker.update([], _ts(i))
    assert tracker.take_lost() == []


def test_take_lost_drains_the_registry() -> None:
    """Prevents unbounded growth over a long drive (NFR-REL-07)."""
    tracker = Tracker(min_hits=1, max_age=1)
    for i in range(20):
        tracker.update([_det(i, (i * 40, 100, i * 40 + 20, 120))], _ts(i))
    for i in range(20, 30):
        tracker.update([], _ts(i))
    tracker.take_lost()
    assert tracker.take_lost() == []


# --------------------------------------------------------------------------
# Bookkeeping — FR-TRK-06
# --------------------------------------------------------------------------


def test_peak_confidence_is_retained() -> None:
    """Severity uses the best view of the defect, not the last."""
    tracker = Tracker(min_hits=1)
    tracker.update([_det(0, (100, 100, 150, 150), conf=0.55)], _ts(0))
    tracker.update([_det(1, (100, 101, 150, 151), conf=0.93)], _ts(1))
    tracker.update([_det(2, (100, 102, 150, 152), conf=0.61)], _ts(2))
    assert tracker.active[0].peak_confidence == pytest.approx(0.93)


def test_frames_seen_and_timestamps() -> None:
    tracker = Tracker(min_hits=1)
    for i in range(5):
        tracker.update([_det(i, (100, 100 + i, 150, 150 + i))], _ts(i))
    track = tracker.active[0]
    assert track.frames_seen == 5
    assert track.first_seen == _ts(0)
    assert track.last_seen == _ts(4)


def test_capture_guard_defaults_false() -> None:
    """At most one capture per track per session (FR-TRK-05)."""
    tracker = Tracker(min_hits=1)
    tracker.update([_det(0, (100, 100, 150, 150))], _ts(0))
    assert tracker.active[0].has_been_captured is False


def test_empty_update_on_empty_tracker_is_safe() -> None:
    tracker = Tracker()
    assert tracker.update([], _ts(0)) == []
