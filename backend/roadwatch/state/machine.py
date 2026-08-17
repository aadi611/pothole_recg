"""Detection state machine (FR-TRK-03).

``UNKNOWN → DETECTED → APPROACHING → VERY_NEAR → CAPTURED → REPORTED``, with
``LOST`` reachable from anywhere.

Transitions are driven by confirmation counts and **distance thresholds** —
never by mere presence in a frame (FR-PRX-01). That rule is enforced here
rather than left to callers: ``advance()`` will not move a track to
APPROACHING or VERY_NEAR without a distance estimate, so there is no code path
where a pothole visible far ahead triggers an alert.

Every transition is logged with the value that caused it (NFR-OBS-02). Tuning
thresholds on real footage means reading those logs afterwards.

Pure logic, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

from roadwatch.core.contracts import DistanceEstimate, TrackState
from roadwatch.core.logging import get_logger

log = get_logger(__name__)


# Terminal states cannot transition further within a session.
_TERMINAL: frozenset[TrackState] = frozenset({TrackState.REPORTED, TrackState.LOST})

# Every legal edge. Anything absent here is a bug, not a silent no-op.
_ALLOWED: dict[TrackState, frozenset[TrackState]] = {
    TrackState.UNKNOWN: frozenset({TrackState.DETECTED, TrackState.LOST}),
    TrackState.DETECTED: frozenset(
        {TrackState.APPROACHING, TrackState.VERY_NEAR, TrackState.LOST}
    ),
    TrackState.APPROACHING: frozenset(
        # Back to DETECTED is legal: distance smoothing can move an estimate
        # back above the warn threshold, and a track that recedes should not
        # stay armed.
        {TrackState.VERY_NEAR, TrackState.DETECTED, TrackState.LOST}
    ),
    TrackState.VERY_NEAR: frozenset(
        {TrackState.CAPTURED, TrackState.APPROACHING, TrackState.LOST}
    ),
    TrackState.CAPTURED: frozenset({TrackState.REPORTED, TrackState.LOST}),
    TrackState.REPORTED: frozenset(),
    TrackState.LOST: frozenset(),
}


@dataclass(frozen=True)
class Transition:
    """A state change and why it happened."""

    track_id: str
    from_state: TrackState
    to_state: TrackState
    trigger: str
    """Human-readable cause, including the value that crossed the threshold."""
    distance_m: float | None = None


class InvalidTransitionError(Exception):
    """An illegal state change was attempted.

    Raised rather than ignored: a pipeline that silently skips an impossible
    transition hides the bug that produced it.
    """


@dataclass(frozen=True)
class StateThresholds:
    """Distance thresholds, in metres, from ``config/proximity.yaml``."""

    warn: float
    capture: float
    min_distance_confidence: float

    def __post_init__(self) -> None:
        if self.warn <= self.capture:
            raise ValueError(
                f"warn ({self.warn}) must exceed capture ({self.capture})"
            )


class StateMachine:
    """Computes the next state for a track.

    Holds no per-track state itself — the state lives on the track. This keeps
    it a pure function of (current state, distance, confirmation), which is
    what makes it testable against synthetic sequences (NFR-TST-01).
    """

    def __init__(self, thresholds: StateThresholds) -> None:
        self._t = thresholds

    @staticmethod
    def can_transition(from_state: TrackState, to_state: TrackState) -> bool:
        return to_state in _ALLOWED.get(from_state, frozenset())

    def advance(
        self,
        *,
        track_id: str,
        current: TrackState,
        is_confirmed: bool,
        distance: DistanceEstimate | None,
    ) -> Transition | None:
        """Return the transition this track should make, or None to stay put.

        Args:
            track_id: For log correlation.
            current: The track's present state.
            is_confirmed: Whether the tracker has seen enough hits
                (FR-TRK-02).
            distance: Latest smoothed estimate, or None when unavailable.

        Returns:
            The transition to apply, or None.
        """
        if current in _TERMINAL:
            return None

        if current is TrackState.UNKNOWN:
            if is_confirmed:
                return self._make(
                    track_id, current, TrackState.DETECTED,
                    trigger="confirmed by tracker (min_hits reached)",
                )
            return None

        # Beyond DETECTED, every transition needs a distance. Without one the
        # track stays where it is: presence in frame is not a reason to alert
        # (FR-PRX-01).
        if distance is None:
            return None

        # A low-confidence estimate may still warn, but must not drive capture
        # (FR-PRX-06). Capture writes evidence and mints a report; doing that
        # on an unreliable distance would put a fabricated measurement into a
        # civic complaint.
        capture_allowed = distance.confidence >= self._t.min_distance_confidence
        d = distance.distance_m

        if current is TrackState.DETECTED:
            if d <= self._t.capture and capture_allowed:
                return self._make(
                    track_id, current, TrackState.VERY_NEAR,
                    trigger=f"distance {d:.1f}m <= capture {self._t.capture}m",
                    distance_m=d,
                )
            if d <= self._t.warn:
                return self._make(
                    track_id, current, TrackState.APPROACHING,
                    trigger=f"distance {d:.1f}m <= warn {self._t.warn}m",
                    distance_m=d,
                )
            return None

        if current is TrackState.APPROACHING:
            if d <= self._t.capture and capture_allowed:
                return self._make(
                    track_id, current, TrackState.VERY_NEAR,
                    trigger=f"distance {d:.1f}m <= capture {self._t.capture}m",
                    distance_m=d,
                )
            if d > self._t.warn:
                return self._make(
                    track_id, current, TrackState.DETECTED,
                    trigger=f"distance {d:.1f}m receded above warn {self._t.warn}m",
                    distance_m=d,
                )
            return None

        # VERY_NEAR and CAPTURED advance on side effects completing, not on
        # distance: the evidence subsystem calls mark_captured/mark_reported.
        return None

    def mark_captured(self, *, track_id: str, current: TrackState) -> Transition:
        """VERY_NEAR → CAPTURED, once evidence is actually on disk."""
        return self._make(
            track_id, current, TrackState.CAPTURED,
            trigger="evidence written to local storage",
        )

    def mark_reported(self, *, track_id: str, current: TrackState) -> Transition:
        """CAPTURED → REPORTED, once the report row exists locally."""
        return self._make(
            track_id, current, TrackState.REPORTED,
            trigger="report persisted to local queue",
        )

    def mark_lost(self, *, track_id: str, current: TrackState, misses: int) -> Transition:
        return self._make(
            track_id, current, TrackState.LOST,
            trigger=f"unmatched for {misses} frames",
        )

    def _make(
        self,
        track_id: str,
        from_state: TrackState,
        to_state: TrackState,
        *,
        trigger: str,
        distance_m: float | None = None,
    ) -> Transition:
        if not self.can_transition(from_state, to_state):
            raise InvalidTransitionError(
                f"track {track_id}: {from_state.value} -> {to_state.value} "
                f"is not a legal transition (trigger: {trigger})"
            )

        transition = Transition(
            track_id=track_id,
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            distance_m=distance_m,
        )

        # The primary tool for tuning thresholds against real footage
        # (NFR-OBS-02).
        log.info(
            "state_transition",
            track_id=track_id,
            from_state=from_state.value,
            to_state=to_state.value,
            trigger=trigger,
            distance_m=round(distance_m, 2) if distance_m is not None else None,
        )
        return transition
