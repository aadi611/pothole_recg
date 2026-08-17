"""State machine tests (FR-TRK-03, FR-PRX-01, FR-PRX-06).

The headline test here is ``test_presence_in_frame_never_triggers_alert``:
FR-PRX-01 says alerts must never fire simply because a pothole appears in
frame, and Phase 2's exit criteria verify it against a video where a pothole
is visible far ahead for many seconds. This asserts the same property at the
unit level.
"""

from __future__ import annotations

import pytest

from roadwatch.core.contracts import DistanceEstimate, TrackState
from roadwatch.state.machine import (
    InvalidTransitionError,
    StateMachine,
    StateThresholds,
)

THRESHOLDS = StateThresholds(warn=10.0, capture=4.0, min_distance_confidence=0.4)


def _machine() -> StateMachine:
    return StateMachine(THRESHOLDS)


def _dist(metres: float, *, confidence: float = 0.8) -> DistanceEstimate:
    return DistanceEstimate(
        distance_m=metres, confidence=confidence, method="planar", assumptions=["flat road"]
    )


# --------------------------------------------------------------------------
# FR-PRX-01 — the core safety property
# --------------------------------------------------------------------------


def test_presence_in_frame_never_triggers_alert() -> None:
    """A confirmed detection with no distance estimate must not advance.

    This is the difference between a useful warning system and one that
    shouts at the driver every time tarmac looks dark.
    """
    m = _machine()
    for _ in range(100):
        transition = m.advance(
            track_id="t1",
            current=TrackState.DETECTED,
            is_confirmed=True,
            distance=None,
        )
        assert transition is None


def test_distant_pothole_visible_for_many_frames_does_not_advance() -> None:
    """Visible far ahead for seconds: still no alert until it is close."""
    m = _machine()
    state = TrackState.DETECTED
    for _ in range(200):
        t = m.advance(
            track_id="t1", current=state, is_confirmed=True, distance=_dist(35.0)
        )
        assert t is None
    assert state is TrackState.DETECTED


def test_advances_only_when_distance_crosses_the_threshold() -> None:
    m = _machine()
    assert m.advance(
        track_id="t1", current=TrackState.DETECTED, is_confirmed=True, distance=_dist(10.1)
    ) is None

    t = m.advance(
        track_id="t1", current=TrackState.DETECTED, is_confirmed=True, distance=_dist(9.9)
    )
    assert t is not None
    assert t.to_state is TrackState.APPROACHING


# --------------------------------------------------------------------------
# FR-PRX-06 — low confidence warns but does not capture
# --------------------------------------------------------------------------


def test_low_confidence_distance_warns_but_does_not_capture() -> None:
    """Below min_distance_confidence, capture is withheld.

    Capture mints a report that may reach a civic authority. Doing that on an
    unreliable distance would put a fabricated measurement into a complaint.
    """
    m = _machine()
    t = m.advance(
        track_id="t1",
        current=TrackState.DETECTED,
        is_confirmed=True,
        distance=_dist(2.0, confidence=0.2),
    )
    assert t is not None
    assert t.to_state is TrackState.APPROACHING, "should warn, not capture"


def test_high_confidence_close_distance_captures() -> None:
    m = _machine()
    t = m.advance(
        track_id="t1",
        current=TrackState.DETECTED,
        is_confirmed=True,
        distance=_dist(2.0, confidence=0.9),
    )
    assert t is not None
    assert t.to_state is TrackState.VERY_NEAR


def test_confidence_exactly_at_threshold_allows_capture() -> None:
    m = _machine()
    t = m.advance(
        track_id="t1",
        current=TrackState.DETECTED,
        is_confirmed=True,
        distance=_dist(2.0, confidence=0.4),
    )
    assert t is not None
    assert t.to_state is TrackState.VERY_NEAR


# --------------------------------------------------------------------------
# FR-TRK-03 — the specified path
# --------------------------------------------------------------------------


def test_unknown_advances_only_when_confirmed() -> None:
    m = _machine()
    assert m.advance(
        track_id="t1", current=TrackState.UNKNOWN, is_confirmed=False, distance=None
    ) is None

    t = m.advance(
        track_id="t1", current=TrackState.UNKNOWN, is_confirmed=True, distance=None
    )
    assert t is not None
    assert t.to_state is TrackState.DETECTED


def test_full_lifecycle() -> None:
    m = _machine()
    state = TrackState.UNKNOWN

    t = m.advance(track_id="t1", current=state, is_confirmed=True, distance=None)
    assert t and t.to_state is TrackState.DETECTED
    state = t.to_state

    t = m.advance(track_id="t1", current=state, is_confirmed=True, distance=_dist(8.0))
    assert t and t.to_state is TrackState.APPROACHING
    state = t.to_state

    t = m.advance(track_id="t1", current=state, is_confirmed=True, distance=_dist(3.0))
    assert t and t.to_state is TrackState.VERY_NEAR
    state = t.to_state

    t = m.mark_captured(track_id="t1", current=state)
    assert t.to_state is TrackState.CAPTURED
    state = t.to_state

    t = m.mark_reported(track_id="t1", current=state)
    assert t.to_state is TrackState.REPORTED


def test_detected_can_skip_straight_to_very_near() -> None:
    """A pothole first detected at close range must not be missed."""
    m = _machine()
    t = m.advance(
        track_id="t1", current=TrackState.DETECTED, is_confirmed=True, distance=_dist(2.0)
    )
    assert t is not None
    assert t.to_state is TrackState.VERY_NEAR


def test_receding_track_returns_to_detected() -> None:
    """Smoothing can push an estimate back out; the track should disarm."""
    m = _machine()
    t = m.advance(
        track_id="t1", current=TrackState.APPROACHING, is_confirmed=True, distance=_dist(15.0)
    )
    assert t is not None
    assert t.to_state is TrackState.DETECTED


def test_terminal_states_do_not_advance() -> None:
    m = _machine()
    for state in (TrackState.REPORTED, TrackState.LOST):
        assert m.advance(
            track_id="t1", current=state, is_confirmed=True, distance=_dist(1.0)
        ) is None


# --------------------------------------------------------------------------
# Legality
# --------------------------------------------------------------------------


def test_illegal_transition_raises_rather_than_silently_skipping() -> None:
    m = _machine()
    with pytest.raises(InvalidTransitionError, match="not a legal transition"):
        m.mark_reported(track_id="t1", current=TrackState.DETECTED)


def test_can_transition_matrix() -> None:
    assert StateMachine.can_transition(TrackState.UNKNOWN, TrackState.DETECTED)
    assert StateMachine.can_transition(TrackState.VERY_NEAR, TrackState.CAPTURED)
    assert not StateMachine.can_transition(TrackState.UNKNOWN, TrackState.CAPTURED)
    assert not StateMachine.can_transition(TrackState.REPORTED, TrackState.DETECTED)


def test_lost_is_reachable_from_every_live_state() -> None:
    for state in (
        TrackState.UNKNOWN,
        TrackState.DETECTED,
        TrackState.APPROACHING,
        TrackState.VERY_NEAR,
        TrackState.CAPTURED,
    ):
        assert StateMachine.can_transition(state, TrackState.LOST)


def test_transition_records_its_trigger() -> None:
    """NFR-OBS-02: the value that caused the change is the tuning signal."""
    m = _machine()
    t = m.advance(
        track_id="t1", current=TrackState.DETECTED, is_confirmed=True, distance=_dist(7.5)
    )
    assert t is not None
    assert "7.5" in t.trigger
    assert t.distance_m == pytest.approx(7.5)


def test_thresholds_reject_inverted_config() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        StateThresholds(warn=3.0, capture=10.0, min_distance_confidence=0.4)
