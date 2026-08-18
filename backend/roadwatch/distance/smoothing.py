"""Distance smoothing (FR-PRX-07).

Raw per-frame estimates jitter, because they inherit the detector's box
jitter: a box edge moving two pixels near the horizon can swing the distance
by metres. Unsmoothed, that causes two visible problems:

1. The on-screen readout flickers, which the driver reads as unreliable.
2. Thresholds **flap** — an estimate oscillating around 10 m would cross the
   warn threshold repeatedly, and each crossing is an alert. The alert
   cooldown would suppress the repeats, but the state machine would still
   thrash between APPROACHING and DETECTED.

Smoothing is per-track: each pothole has its own history, and mixing two
potholes' distances would be worse than not smoothing at all.

Pure logic, no I/O.
"""

from __future__ import annotations

from collections import deque

from roadwatch.core.contracts import DistanceEstimate


class DistanceSmoother:
    """Per-track moving average over recent distance estimates.

    A plain moving average rather than a Kalman filter: the underlying signal
    is a vehicle approaching at roughly constant speed, the measurement noise
    is not Gaussian (it is quantisation plus detector jitter), and a filter
    tuned on assumptions we cannot verify would be false precision. A window
    is easy to reason about and easy to check.

    Args:
        window: Frames to average over. Larger is smoother but lags, and lag
            here means warning late. Five frames at ~15 fps is a third of a
            second.
    """

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._window = window
        self._history: dict[str, deque[float]] = {}
        self._confidence: dict[str, deque[float]] = {}

    def update(self, track_id: str, estimate: DistanceEstimate) -> DistanceEstimate:
        """Fold a new estimate into the track's history.

        Infinite distances (at or above the horizon) are not averaged — they
        would poison the mean. They are passed through unchanged and do not
        disturb the stored history.
        """
        if not _is_finite(estimate.distance_m):
            return estimate

        distances = self._history.setdefault(track_id, deque(maxlen=self._window))
        confidences = self._confidence.setdefault(track_id, deque(maxlen=self._window))
        distances.append(estimate.distance_m)
        confidences.append(estimate.confidence)

        smoothed = sum(distances) / len(distances)

        # Recompute time-to-pothole from the smoothed distance, so the two
        # numbers shown to the driver stay consistent with each other.
        time_to_pothole = None
        if estimate.time_to_pothole_s is not None and estimate.distance_m > 0:
            speed_ms = estimate.distance_m / estimate.time_to_pothole_s
            if speed_ms > 0:
                time_to_pothole = round(smoothed / speed_ms, 2)

        return DistanceEstimate(
            distance_m=round(smoothed, 2),
            time_to_pothole_s=time_to_pothole,
            # Average confidence too: one good frame among four poor ones
            # should not present as a confident estimate.
            confidence=round(sum(confidences) / len(confidences), 3),
            method=estimate.method,
            assumptions=[
                *estimate.assumptions,
                f"smoothed over {len(distances)} frames",
            ],
        )

    def forget(self, track_id: str) -> None:
        """Drop a track's history.

        Called when a track is LOST. Without this the dictionaries grow for
        the whole drive (NFR-REL-07).
        """
        self._history.pop(track_id, None)
        self._confidence.pop(track_id, None)

    def retain_only(self, track_ids: set[str]) -> None:
        """Drop history for every track not in ``track_ids``."""
        for known in list(self._history):
            if known not in track_ids:
                self.forget(known)

    def __len__(self) -> int:
        return len(self._history)


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
