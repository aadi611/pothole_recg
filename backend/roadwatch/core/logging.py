"""Structured logging (NFR-OBS-01).

JSON to stdout, with correlation fields bound once and carried through:
``session_id``, ``device_id``, ``track_id``, ``report_id``. Tuning thresholds
on real footage means reading these logs afterwards, so state-machine
transitions log the value that triggered them (NFR-OBS-02).

Never log tokens, passwords, or full GPS traces (NFR-SEC-08).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Install the structlog pipeline. Call once at startup.

    Args:
        level: Standard logging level name.
        json_output: JSON for production and log analysis; set False for a
            human-readable console during development.
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    processors.append(
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a logger bound to ``name``."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def bind_session(**fields: Any) -> None:
    """Bind correlation fields for the current context.

    Everything logged afterwards in this context carries them, so a drive can
    be reconstructed from the log alone.
    """
    structlog.contextvars.bind_contextvars(**fields)


def clear_session() -> None:
    """Clear bound correlation fields."""
    structlog.contextvars.clear_contextvars()
