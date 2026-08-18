"""Driver alerting (FR-ALT-01..06).

The governing constraint here is not technical. **The driver's attention is
the safety-critical resource** (ethical guardrail 4). Every alert costs some
of it, so this module is mostly about *not* alerting:

- One pothole gets at most one ``warn`` and at most one ``near``, ever
  (FR-ALT-02). A repeat-firing alert is worse than none, because the driver
  learns to tune it out.
- A global minimum gap between alerts, across all tracks. On a badly broken
  road the detector finds defects continuously; without this the system
  becomes a rattle.
- Silence below a configurable speed (FR-ALT-05): parked or crawling in
  traffic needs no warnings.
- Alerts are driven only by distance or time-to-impact that the state machine
  has already gated (FR-PRX-01).

Sink failures never propagate. A missing audio device must not stop the
pipeline (FR-ALT-03, NFR-REL-04).

Pure logic apart from the sinks, so cooldown behaviour is testable against a
synthetic event stream (NFR-TST-01).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from roadwatch.core.logging import get_logger

if TYPE_CHECKING:
    from roadwatch.core.config import AlertsConfig
    from roadwatch.core.contracts import DistanceEstimate

log = get_logger(__name__)

AlertLevel = Literal["warn", "near"]


@dataclass(frozen=True)
class Alert:
    """One alert that was actually issued."""

    level: AlertLevel
    track_id: str
    text: str
    distance_m: float | None
    issued_at: float
    """``time.monotonic()`` — used for gap arithmetic, not wall-clock display."""


@runtime_checkable
class AlertSink(Protocol):
    """A way of telling the driver. Screen, tone, TTS, haptics."""

    name: str

    def emit(self, alert: Alert) -> None:
        """Deliver the alert. Must not block the pipeline."""
        ...

    def close(self) -> None: ...


class AlertEngine:
    """Decides whether to alert, and dispatches to sinks.

    Args:
        config: Validated ``config/alerts.yaml``.
        sinks: Output channels. An empty list is legal — the engine still
            enforces cooldowns and records what it would have issued, which
            is what the tests exercise.
        clock: Monotonic time source, injectable for deterministic tests.
    """

    def __init__(
        self,
        config: AlertsConfig,
        sinks: list[AlertSink] | None = None,
        *,
        clock: object = time.monotonic,
    ) -> None:
        self._config = config
        self._sinks = sinks or []
        self._clock = clock

        # Per-track record of which levels have already fired. This is what
        # makes "one warn, one near, ever" true (FR-ALT-02).
        self._fired: dict[str, set[AlertLevel]] = {}
        self._last_alert_at: float | None = None
        self.issued: list[Alert] = []

    def _now(self) -> float:
        return float(self._clock())  # type: ignore[operator]

    def _text_for(self, level: AlertLevel, distance_m: float | None) -> str:
        tts = self._config.tts
        if level == "warn":
            if distance_m is None:
                return "Pothole ahead"
            return tts.warn_template.format(distance_m=distance_m)
        return tts.near_template

    def should_alert(
        self,
        *,
        track_id: str,
        level: AlertLevel,
        speed_kmh: float | None,
    ) -> tuple[bool, str]:
        """Whether to fire, and why not if not.

        Returned as a reason string rather than a bare bool so suppression is
        visible in the logs — tuning alert behaviour on real footage means
        knowing which rule silenced what.
        """
        if speed_kmh is not None and speed_kmh < self._config.suppress_below_speed_kmh:
            return False, (
                f"speed {speed_kmh:.1f} km/h below "
                f"{self._config.suppress_below_speed_kmh} km/h threshold"
            )

        if level in self._fired.get(track_id, set()):
            return False, f"{level} already issued for this track"

        now = self._now()
        if self._last_alert_at is not None:
            gap = now - self._last_alert_at
            if gap < self._config.global_min_gap_s and not self._is_escalation(
                track_id, level
            ):
                return False, (
                    f"only {gap:.2f}s since the last alert, "
                    f"minimum gap is {self._config.global_min_gap_s}s"
                )

        return True, "ok"

    def _is_escalation(self, track_id: str, level: AlertLevel) -> bool:
        """Whether this is the same pothole getting closer.

        The global gap exists to stop *unrelated* alerts rattling. It must not
        silence an escalation on a track that already warned: `near` fires at
        the capture threshold, metres from impact, and is the most urgent
        thing this system says.

        At 40 km/h the run from the 10 m warn threshold to the 4 m capture
        threshold takes about 0.5 s — well inside a 1.5 s gap — so without
        this exemption the `near` alert would essentially always be
        suppressed, defeating the two-level escalation FR-ALT-01 specifies.
        """
        return level == "near" and "warn" in self._fired.get(track_id, set())

    def alert(
        self,
        *,
        track_id: str,
        level: AlertLevel,
        distance: DistanceEstimate | None,
        speed_kmh: float | None,
    ) -> Alert | None:
        """Issue an alert if the rules permit. Returns it, or None."""
        distance_m = distance.distance_m if distance is not None else None

        permitted, reason = self.should_alert(
            track_id=track_id, level=level, speed_kmh=speed_kmh
        )
        if not permitted:
            log.debug(
                "alert_suppressed", track_id=track_id, level=level, reason=reason
            )
            return None

        now = self._now()
        alert = Alert(
            level=level,
            track_id=track_id,
            text=self._text_for(level, distance_m),
            distance_m=distance_m,
            issued_at=now,
        )

        self._fired.setdefault(track_id, set()).add(level)
        self._last_alert_at = now
        self.issued.append(alert)

        log.info(
            "alert_issued",
            track_id=track_id,
            level=level,
            distance_m=round(distance_m, 1) if distance_m is not None else None,
            text=alert.text,
        )

        self._dispatch(alert)
        return alert

    def _dispatch(self, alert: Alert) -> None:
        """Send to every sink, isolating failures.

        A sink that raises must not take the pipeline with it: no audio device
        is a degraded experience, not a crash (FR-ALT-03, NFR-REL-04).
        """
        for sink in self._sinks:
            started = self._now()
            try:
                sink.emit(alert)
            except Exception as exc:
                log.error(
                    "alert_sink_failed",
                    sink=sink.name,
                    track_id=alert.track_id,
                    error=str(exc),
                )
                continue

            elapsed_ms = (self._now() - started) * 1000
            if elapsed_ms > self._config.latency_budget_ms:
                # Logged, not raised: a slow sink is a tuning problem, and
                # crashing over it would be worse than the delay (FR-ALT-06).
                log.warning(
                    "alert_latency_exceeded",
                    sink=sink.name,
                    elapsed_ms=round(elapsed_ms, 1),
                    budget_ms=self._config.latency_budget_ms,
                )

    def forget(self, track_id: str) -> None:
        """Drop a track's alert history once it is LOST."""
        self._fired.pop(track_id, None)

    def retain_only(self, track_ids: set[str]) -> None:
        """Bound memory across a long drive (NFR-REL-07)."""
        for known in list(self._fired):
            if known not in track_ids:
                self._fired.pop(known, None)

    def close(self) -> None:
        for sink in self._sinks:
            try:
                sink.close()
            except Exception as exc:
                log.error("alert_sink_close_failed", sink=sink.name, error=str(exc))
