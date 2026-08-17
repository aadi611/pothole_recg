"""Multi-object tracking (FR-TRK-01, FR-TRK-02, FR-TRK-04).

Greedy IoU matching with a constant-velocity Kalman filter per track. No
re-identification model: a forward-facing vehicle camera sees each defect for
a second or two on a predictable trajectory (growing and moving down the
frame), so appearance features would cost inference budget to solve a problem
we do not have.

**Pure logic, no I/O** (architecture rule): every function here is a
deterministic function of its inputs, unit-testable with synthetic detection
sequences and no camera.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from roadwatch.core.contracts import BBox, Detection, HazardClass, TrackState
from roadwatch.core.logging import get_logger

log = get_logger(__name__)


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two boxes. 0.0 when disjoint."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0

    union = a.area_px + b.area_px - intersection
    return intersection / union if union > 0 else 0.0


class _KalmanBox:
    """Constant-velocity filter over ``[cx, cy, w, h]``.

    Deliberately simple. The measurement is a bounding box; the state adds
    velocity so a track survives a frame or two of missed detection without
    its predicted position drifting off the defect.

    Implemented directly rather than via filterpy to avoid a dependency for
    ~40 lines of linear algebra, and to keep this module import-light — it is
    on the per-frame hot path.
    """

    def __init__(self, box: BBox) -> None:
        # State: [cx, cy, w, h, vx, vy]. Box size is not given a velocity: a
        # pothole's apparent size grows non-linearly with approach, and a
        # linear size model diverges badly in the last few metres.
        self.x = np.array(
            [
                (box.x1 + box.x2) / 2,
                (box.y1 + box.y2) / 2,
                box.width,
                box.height,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        self.P = np.eye(6) * 10.0
        self.P[4:, 4:] *= 100.0  # velocity starts highly uncertain

        self.F = np.eye(6)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0

        self.H = np.zeros((4, 6))
        self.H[:4, :4] = np.eye(4)

        self.Q = np.eye(6) * 0.05
        self.Q[4:, 4:] *= 0.5
        self.R = np.eye(4) * 1.0

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, box: BBox) -> None:
        z = np.array(
            [
                (box.x1 + box.x2) / 2,
                (box.y1 + box.y2) / 2,
                box.width,
                box.height,
            ],
            dtype=np.float64,
        )
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def to_bbox(self) -> BBox:
        cx, cy, w, h = self.x[:4]
        w = max(1.0, float(w))
        h = max(1.0, float(h))
        return BBox(x1=cx - w / 2, y1=cy - h / 2, x2=cx + w / 2, y2=cy + h / 2)


@dataclass
class TrackedObject:
    """One physical defect followed across frames.

    Carries the bookkeeping severity and report quality depend on
    (FR-TRK-06): peak confidence, frames seen, first/last seen.
    """

    track_id: str
    hazard_class: HazardClass
    state: TrackState
    first_seen: datetime
    last_seen: datetime
    latest: Detection
    frames_seen: int = 1
    peak_confidence: float = 0.0
    hits: int = 1
    """Total detections matched to this track."""
    consecutive_misses: int = 0
    has_been_captured: bool = False
    """Set once evidence is captured. Enforces at most one capture per track
    per session (FR-TRK-05)."""
    has_been_reported: bool = False
    _kf: _KalmanBox | None = field(default=None, repr=False)

    @property
    def predicted_bbox(self) -> BBox:
        return self._kf.to_bbox() if self._kf is not None else self.latest.bbox


class Tracker:
    """Greedy IoU tracker with Kalman prediction.

    Defaults are measured, not guessed — see ``scripts/bench_tracker.py``,
    which sweeps them against simulated detector noise:

    ==========================  ==========  ==========
    config                      continuity  separation
    ==========================  ==========  ==========
    iou=0.30 max_age=10                54%         86%
    iou=0.20 max_age=10                80%         98%
    iou=0.15 max_age=15                90%        100%
    ==========================  ==========  ==========

    *continuity* = one approaching pothole yielding exactly one track id under
    15% missed frames and 6px box jitter (the Phase 1 exit criterion is >=80%).
    *separation* = two distinct potholes staying two tracks.

    A tighter IoU gate is worse on **both** counts, which is initially
    counter-intuitive. The reason is that the Kalman filter already predicts
    where the box should be, so the gate is not protecting against
    displacement — it is only rejecting the detector's own jitter. Tightening
    it fragments one pothole into several tracks, and each fragment is a fresh
    unconfirmed track that can then mis-associate.

    Args:
        iou_threshold: Minimum IoU to associate a detection with a track.
        max_age: Frames a track survives unmatched before going LOST
            (FR-TRK-04).
        min_hits: Detections required before a track is promoted past
            UNKNOWN, suppressing single-frame false positives (FR-TRK-02).
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.15,
        max_age: int = 15,
        min_hits: int = 3,
    ) -> None:
        if not 0.0 < iou_threshold < 1.0:
            raise ValueError(f"iou_threshold must be in (0,1), got {iou_threshold}")
        if min_hits < 1:
            raise ValueError(f"min_hits must be >= 1, got {min_hits}")

        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._min_hits = min_hits
        self._tracks: dict[str, TrackedObject] = {}
        self._lost: dict[str, TrackedObject] = {}

    @property
    def active(self) -> list[TrackedObject]:
        """Tracks currently being followed."""
        return list(self._tracks.values())

    @property
    def confirmed(self) -> list[TrackedObject]:
        """Tracks that have passed the confirmation threshold (FR-TRK-02)."""
        return [t for t in self._tracks.values() if t.hits >= self._min_hits]

    def _match(
        self, detections: list[Detection]
    ) -> tuple[dict[str, Detection], list[Detection]]:
        """Associate detections to tracks, greedily by descending IoU.

        Greedy rather than Hungarian: with the handful of simultaneous defects
        a road camera sees, the optimal assignment and the greedy one are
        almost always identical, and greedy is far cheaper on the hot path.
        """
        if not self._tracks or not detections:
            return {}, list(detections)

        pairs: list[tuple[float, str, int]] = []
        for track_id, track in self._tracks.items():
            predicted = track.predicted_bbox
            for idx, det in enumerate(detections):
                # Never merge two different hazard types into one track.
                if det.hazard_class != track.hazard_class:
                    continue
                score = iou(predicted, det.bbox)
                if score >= self._iou_threshold:
                    pairs.append((score, track_id, idx))

        pairs.sort(key=lambda p: p[0], reverse=True)

        matched: dict[str, Detection] = {}
        used_tracks: set[str] = set()
        used_dets: set[int] = set()
        for _score, track_id, idx in pairs:
            if track_id in used_tracks or idx in used_dets:
                continue
            matched[track_id] = detections[idx]
            used_tracks.add(track_id)
            used_dets.add(idx)

        unmatched = [d for i, d in enumerate(detections) if i not in used_dets]
        return matched, unmatched

    def update(self, detections: list[Detection], timestamp: datetime) -> list[TrackedObject]:
        """Advance the tracker by one processed frame.

        Args:
            detections: Detections for this frame. May be empty.
            timestamp: Frame capture time.

        Returns:
            Currently active tracks.
        """
        for track in self._tracks.values():
            if track._kf is not None:
                track._kf.predict()

        matched, unmatched = self._match(detections)

        for track_id, det in matched.items():
            track = self._tracks[track_id]
            if track._kf is not None:
                track._kf.update(det.bbox)
            track.latest = det
            track.last_seen = timestamp
            track.frames_seen += 1
            track.hits += 1
            track.consecutive_misses = 0
            track.peak_confidence = max(track.peak_confidence, det.confidence)

            if track.state is TrackState.UNKNOWN and track.hits >= self._min_hits:
                track.state = TrackState.DETECTED
                log.info(
                    "track_confirmed",
                    track_id=track_id,
                    hits=track.hits,
                    trigger=f"hits >= min_hits ({self._min_hits})",
                    peak_confidence=round(track.peak_confidence, 3),
                )

        for track_id in list(self._tracks):
            if track_id not in matched:
                track = self._tracks[track_id]
                track.consecutive_misses += 1
                if track.consecutive_misses > self._max_age:
                    track.state = TrackState.LOST
                    log.info(
                        "track_lost",
                        track_id=track_id,
                        trigger=f"unmatched for {track.consecutive_misses} frames",
                        frames_seen=track.frames_seen,
                        was_captured=track.has_been_captured,
                    )
                    # A LOST track that already captured still owes a report
                    # (FR-TRK-04), so it is retained, not discarded.
                    self._lost[track_id] = track
                    del self._tracks[track_id]

        for det in unmatched:
            track_id = str(uuid.uuid4())
            self._tracks[track_id] = TrackedObject(
                track_id=track_id,
                hazard_class=det.hazard_class,
                state=TrackState.UNKNOWN,
                first_seen=timestamp,
                last_seen=timestamp,
                latest=det,
                peak_confidence=det.confidence,
                _kf=_KalmanBox(det.bbox),
            )

        return self.active

    def take_lost(self) -> list[TrackedObject]:
        """Remove and return LOST tracks that still owe a report.

        The caller is responsible for completing them (FR-TRK-04). Draining
        here keeps the registry from growing over a long drive (NFR-REL-07).
        """
        pending = [t for t in self._lost.values() if t.has_been_captured and not t.has_been_reported]
        self._lost.clear()
        return pending

    def reset(self) -> None:
        self._tracks.clear()
        self._lost.clear()
