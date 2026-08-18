"""Alert engine tests (FR-ALT-01..06).

Phase 2 requires cooldown logic to be tested against a synthetic event stream
(spec/05_BUILD_PHASES.md), so time is injected rather than slept.

The property that matters most: one pothole produces at most one warn and at
most one near, ever. A repeat-firing alert is worse than no alert, because
the driver learns to ignore it.
"""

from __future__ import annotations

from roadwatch.alerts.engine import Alert, AlertEngine, AlertSink
from roadwatch.alerts.sinks import ConsoleSink, RecordingSink, build_sinks
from roadwatch.core.config import AlertsConfig, TtsConfig
from roadwatch.core.contracts import DistanceEstimate


class FakeClock:
    """Injectable monotonic clock."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _config(**overrides: object) -> AlertsConfig:
    base = {
        "cooldown_s": 6.0,
        "global_min_gap_s": 1.5,
        "sinks": ["recording"],
        "tts": TtsConfig(
            engine="pyttsx3",
            warn_template="Pothole ahead, {distance_m:.0f} meters",
            near_template="Pothole very near",
            rate_wpm=190,
        ),
        "suppress_below_speed_kmh": 5.0,
        "latency_budget_ms": 300,
    }
    base.update(overrides)
    return AlertsConfig(**base)  # type: ignore[arg-type]


def _dist(m: float) -> DistanceEstimate:
    return DistanceEstimate(distance_m=m, confidence=0.8, method="planar")


# --------------------------------------------------------------------------
# FR-ALT-02 — one warn, one near, ever
# --------------------------------------------------------------------------


def test_one_track_gets_at_most_one_warn() -> None:
    clock = FakeClock()
    engine = AlertEngine(_config(), [], clock=clock)

    first = engine.alert(track_id="t1", level="warn", distance=_dist(10), speed_kmh=40)
    assert first is not None

    # Even much later, and even as it gets closer, no second warn.
    for _ in range(20):
        clock.advance(5.0)
        assert engine.alert(track_id="t1", level="warn", distance=_dist(8), speed_kmh=40) is None

    assert len(engine.issued) == 1


def test_a_track_can_warn_then_near() -> None:
    """Two distinct levels are allowed; a repeat of either is not."""
    clock = FakeClock()
    engine = AlertEngine(_config(), [], clock=clock)

    assert engine.alert(track_id="t1", level="warn", distance=_dist(10), speed_kmh=40)
    clock.advance(2.0)
    assert engine.alert(track_id="t1", level="near", distance=_dist(3), speed_kmh=40)
    clock.advance(2.0)
    assert engine.alert(track_id="t1", level="near", distance=_dist(2), speed_kmh=40) is None

    assert [a.level for a in engine.issued] == ["warn", "near"]


def test_different_tracks_alert_independently() -> None:
    clock = FakeClock()
    engine = AlertEngine(_config(), [], clock=clock)

    assert engine.alert(track_id="t1", level="warn", distance=_dist(10), speed_kmh=40)
    clock.advance(2.0)
    assert engine.alert(track_id="t2", level="warn", distance=_dist(10), speed_kmh=40)
    assert len(engine.issued) == 2


# --------------------------------------------------------------------------
# Global gap
# --------------------------------------------------------------------------


def test_global_gap_prevents_an_alert_rattle() -> None:
    """On a badly broken road the detector fires continuously.

    Without a global gap the system becomes a rattle the driver tunes out.
    """
    clock = FakeClock()
    engine = AlertEngine(_config(global_min_gap_s=1.5), [], clock=clock)

    assert engine.alert(track_id="t1", level="warn", distance=_dist(10), speed_kmh=40)
    clock.advance(0.3)
    assert engine.alert(track_id="t2", level="warn", distance=_dist(10), speed_kmh=40) is None
    clock.advance(0.3)
    assert engine.alert(track_id="t3", level="warn", distance=_dist(10), speed_kmh=40) is None

    assert len(engine.issued) == 1


def test_alert_allowed_once_the_gap_has_passed() -> None:
    clock = FakeClock()
    engine = AlertEngine(_config(global_min_gap_s=1.5), [], clock=clock)

    engine.alert(track_id="t1", level="warn", distance=_dist(10), speed_kmh=40)
    clock.advance(1.6)
    assert engine.alert(track_id="t2", level="warn", distance=_dist(10), speed_kmh=40)


def test_ten_potholes_in_quick_succession_do_not_all_fire() -> None:
    clock = FakeClock()
    engine = AlertEngine(_config(global_min_gap_s=1.5), [], clock=clock)

    for i in range(10):
        engine.alert(track_id=f"t{i}", level="warn", distance=_dist(10), speed_kmh=40)
        clock.advance(0.2)

    # 10 tracks over 2 s with a 1.5 s gap allows at most 2.
    assert len(engine.issued) <= 2


# --------------------------------------------------------------------------
# FR-ALT-05 — speed suppression
# --------------------------------------------------------------------------


def test_no_alerts_when_parked() -> None:
    engine = AlertEngine(_config(), [], clock=FakeClock())
    assert engine.alert(track_id="t1", level="warn", distance=_dist(5), speed_kmh=0.0) is None


def test_no_alerts_when_crawling_in_traffic() -> None:
    engine = AlertEngine(_config(suppress_below_speed_kmh=5.0), [], clock=FakeClock())
    assert engine.alert(track_id="t1", level="warn", distance=_dist(5), speed_kmh=3.0) is None


def test_alerts_resume_above_the_speed_threshold() -> None:
    engine = AlertEngine(_config(suppress_below_speed_kmh=5.0), [], clock=FakeClock())
    assert engine.alert(track_id="t1", level="warn", distance=_dist(10), speed_kmh=30.0)


def test_unknown_speed_does_not_suppress() -> None:
    """No GPS must not mean no warnings (NFR-REL-04 degradation)."""
    engine = AlertEngine(_config(), [], clock=FakeClock())
    assert engine.alert(track_id="t1", level="warn", distance=_dist(10), speed_kmh=None)


# --------------------------------------------------------------------------
# FR-ALT-01 — message content
# --------------------------------------------------------------------------


def test_warn_message_includes_the_distance() -> None:
    engine = AlertEngine(_config(), [], clock=FakeClock())
    alert = engine.alert(track_id="t1", level="warn", distance=_dist(10.4), speed_kmh=40)
    assert alert is not None
    assert "10" in alert.text


def test_near_message_does_not_state_a_distance() -> None:
    """At the capture threshold the number is noise; the urgency is the point."""
    engine = AlertEngine(_config(), [], clock=FakeClock())
    alert = engine.alert(track_id="t1", level="near", distance=_dist(3), speed_kmh=40)
    assert alert is not None
    assert alert.text == "Pothole very near"


def test_missing_distance_still_produces_a_message() -> None:
    engine = AlertEngine(_config(), [], clock=FakeClock())
    alert = engine.alert(track_id="t1", level="warn", distance=None, speed_kmh=40)
    assert alert is not None
    assert "Pothole" in alert.text


# --------------------------------------------------------------------------
# FR-ALT-03 — sinks degrade, never crash
# --------------------------------------------------------------------------


class ExplodingSink:
    name = "exploding"

    def emit(self, alert: Alert) -> None:
        raise RuntimeError("no audio device")

    def close(self) -> None:
        raise RuntimeError("still broken")


def test_a_failing_sink_does_not_stop_the_pipeline() -> None:
    """No audio device is a degraded experience, not a crash."""
    engine = AlertEngine(_config(), [ExplodingSink()], clock=FakeClock())
    alert = engine.alert(track_id="t1", level="warn", distance=_dist(10), speed_kmh=40)
    assert alert is not None


def test_a_failing_sink_does_not_block_the_others() -> None:
    recording = RecordingSink()
    engine = AlertEngine(_config(), [ExplodingSink(), recording], clock=FakeClock())
    engine.alert(track_id="t1", level="warn", distance=_dist(10), speed_kmh=40)
    assert len(recording.alerts) == 1


def test_close_survives_a_failing_sink() -> None:
    engine = AlertEngine(_config(), [ExplodingSink()], clock=FakeClock())
    engine.close()


def test_recording_sink_satisfies_the_protocol() -> None:
    assert isinstance(RecordingSink(), AlertSink)
    assert isinstance(ConsoleSink(), AlertSink)


def test_build_sinks_skips_unknown_names() -> None:
    """A config typo must not stop a vehicle detecting potholes."""
    sinks = build_sinks(["screen", "nonsense", "recording"])
    assert len(sinks) == 2


def test_tts_sink_is_constructible_without_audio() -> None:
    """Construction must never raise, whatever the audio situation."""
    from roadwatch.alerts.sinks import TtsSink

    sink = TtsSink()
    sink.emit(
        Alert(level="warn", track_id="t1", text="test", distance_m=10.0, issued_at=0.0)
    )
    sink.close()


# --------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------


def test_suppression_reason_is_reported() -> None:
    """Tuning alerts on real footage means knowing which rule silenced what."""
    engine = AlertEngine(_config(), [], clock=FakeClock())
    permitted, reason = engine.should_alert(track_id="t1", level="warn", speed_kmh=1.0)
    assert not permitted
    assert "speed" in reason


def test_forget_bounds_memory_over_a_long_drive() -> None:
    engine = AlertEngine(_config(global_min_gap_s=0.0), [], clock=FakeClock())
    for i in range(500):
        engine.alert(track_id=f"t{i}", level="warn", distance=_dist(10), speed_kmh=40)
    assert len(engine._fired) == 500
    engine.retain_only({"t1"})
    assert len(engine._fired) == 1


def test_forgotten_track_could_alert_again() -> None:
    """forget() is only called on LOST, so this cannot re-fire live tracks."""
    clock = FakeClock()
    engine = AlertEngine(_config(), [], clock=clock)
    engine.alert(track_id="t1", level="warn", distance=_dist(10), speed_kmh=40)
    engine.forget("t1")
    clock.advance(5.0)
    assert engine.alert(track_id="t1", level="warn", distance=_dist(10), speed_kmh=40)


# --------------------------------------------------------------------------
# Escalation vs the global gap
# --------------------------------------------------------------------------


def test_near_is_not_suppressed_by_the_global_gap() -> None:
    """The most urgent alert must not be silenced by the anti-rattle rule.

    Found by running the wired pipeline: at 40 km/h a pothole takes ~0.5s to
    travel from the 10m warn threshold to the 4m capture threshold, which is
    inside the 1.5s global gap. Without an escalation exemption the `near`
    alert was suppressed every single time, defeating the two-level
    escalation FR-ALT-01 specifies.
    """
    clock = FakeClock()
    engine = AlertEngine(_config(global_min_gap_s=1.5), [], clock=clock)

    assert engine.alert(track_id="t1", level="warn", distance=_dist(10), speed_kmh=40)
    clock.advance(0.5)  # well inside the global gap
    near = engine.alert(track_id="t1", level="near", distance=_dist(3), speed_kmh=40)

    assert near is not None, "near alert was suppressed by the global gap"
    assert [a.level for a in engine.issued] == ["warn", "near"]


def test_escalation_exemption_does_not_apply_to_a_different_track() -> None:
    """The exemption is for one pothole getting closer, not for any near."""
    clock = FakeClock()
    engine = AlertEngine(_config(global_min_gap_s=1.5), [], clock=clock)

    engine.alert(track_id="t1", level="warn", distance=_dist(10), speed_kmh=40)
    clock.advance(0.3)
    other = engine.alert(track_id="t2", level="near", distance=_dist(3), speed_kmh=40)

    assert other is None, "an unrelated track bypassed the global gap"


def test_escalation_exemption_requires_a_prior_warn() -> None:
    """A near with no preceding warn is not an escalation, so the gap applies."""
    clock = FakeClock()
    engine = AlertEngine(_config(global_min_gap_s=1.5), [], clock=clock)

    engine.alert(track_id="t1", level="warn", distance=_dist(10), speed_kmh=40)
    clock.advance(0.3)
    # t2 was never warned, so its near is a fresh alert subject to the gap.
    assert engine.alert(track_id="t2", level="near", distance=_dist(3), speed_kmh=40) is None


def test_near_still_fires_only_once() -> None:
    """The exemption must not reopen repeat-firing (FR-ALT-02)."""
    clock = FakeClock()
    engine = AlertEngine(_config(global_min_gap_s=1.5), [], clock=clock)

    engine.alert(track_id="t1", level="warn", distance=_dist(10), speed_kmh=40)
    clock.advance(0.4)
    assert engine.alert(track_id="t1", level="near", distance=_dist(3), speed_kmh=40)
    for _ in range(10):
        clock.advance(0.2)
        assert engine.alert(track_id="t1", level="near", distance=_dist(2), speed_kmh=40) is None

    assert len(engine.issued) == 2
