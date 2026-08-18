"""Alert output channels (FR-ALT-03).

Every sink degrades silently when its hardware is unavailable. A vehicle with
no audio device still gets on-screen warnings; it does not get a crash
(NFR-REL-04).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from roadwatch.core.logging import get_logger

if TYPE_CHECKING:
    from roadwatch.alerts.engine import Alert

log = get_logger(__name__)


class ConsoleSink:
    """Writes alerts to stderr.

    Always available, which makes it the fallback when nothing else is.
    """

    name = "console"

    def __init__(self, *, stream: object = None) -> None:
        self._stream = stream or sys.stderr

    def emit(self, alert: Alert) -> None:
        marker = "!!" if alert.level == "near" else "!"
        self._stream.write(f"{marker} {alert.text}\n")  # type: ignore[attr-defined]
        self._stream.flush()  # type: ignore[attr-defined]

    def close(self) -> None:
        pass


class TtsSink:
    """Spoken alerts via pyttsx3.

    Construction never raises. If the TTS engine is unavailable — no audio
    device, no driver, a headless box — the sink marks itself inactive and
    every ``emit`` becomes a no-op. The alternative, raising at startup, would
    mean a missing speaker prevents the whole detection pipeline from running.
    """

    name = "tts"

    def __init__(self, *, rate_wpm: int = 190) -> None:
        self._engine: object | None = None
        self.available = False

        try:
            import pyttsx3
        except ImportError:
            log.warning(
                "tts_unavailable",
                reason="pyttsx3 not installed",
                impact="alerts will be visual only",
            )
            return

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", rate_wpm)
        except Exception as exc:
            log.warning(
                "tts_unavailable",
                reason=str(exc),
                impact="alerts will be visual only",
            )
            return

        self._engine = engine
        self.available = True
        log.info("tts_ready", rate_wpm=rate_wpm)

    def emit(self, alert: Alert) -> None:
        if not self.available or self._engine is None:
            return
        # say() queues; runAndWait() blocks until spoken. The engine calls
        # this from the alert dispatch path, which the latency budget check
        # in AlertEngine measures (FR-ALT-06).
        self._engine.say(alert.text)  # type: ignore[attr-defined]
        self._engine.runAndWait()  # type: ignore[attr-defined]

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.stop()  # type: ignore[attr-defined]
            except Exception as exc:
                log.debug("tts_stop_failed", error=str(exc))


class RecordingSink:
    """Captures alerts in memory. For tests and for the session debug log."""

    name = "recording"

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def emit(self, alert: Alert) -> None:
        self.alerts.append(alert)

    def close(self) -> None:
        pass


def build_sinks(names: list[str], *, rate_wpm: int = 190) -> list[object]:
    """Construct the sinks named in config.

    An unknown name is logged and skipped rather than fatal: a typo in the
    sink list should not stop a vehicle from detecting potholes.
    """
    sinks: list[object] = []
    for name in names:
        if name in ("screen", "console"):
            sinks.append(ConsoleSink())
        elif name == "tts":
            sinks.append(TtsSink(rate_wpm=rate_wpm))
        elif name == "recording":
            sinks.append(RecordingSink())
        else:
            log.warning("unknown_alert_sink", name=name, action="skipped")
    return sinks
