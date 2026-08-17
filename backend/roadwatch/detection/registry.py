"""Detector registry (FR-DET-01).

Maps the ``detector`` key in ``config/detection.yaml`` to an implementation.
Adding a detector — an ONNX one for the edge phase (FR-DET-07), say — means
registering it here; no caller changes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from roadwatch.core.errors import ConfigError

if TYPE_CHECKING:
    from roadwatch.core.config import DetectionConfig
    from roadwatch.detection.base import Detector

_BUILDERS: dict[str, Callable[[DetectionConfig], Detector]] = {}


def register(name: str, builder: Callable[[DetectionConfig], Detector]) -> None:
    _BUILDERS[name] = builder


def build(config: DetectionConfig) -> Detector:
    """Construct the detector named in config.

    Raises:
        ConfigError: if the name is not registered, listing what is available.
    """
    builder = _BUILDERS.get(config.detector)
    if builder is None:
        raise ConfigError(
            f"unknown detector '{config.detector}'. "
            f"Registered: {sorted(_BUILDERS) or '(none)'}"
        )
    return builder(config)


def available() -> list[str]:
    return sorted(_BUILDERS)


def _build_yolo(config: DetectionConfig) -> Detector:
    # Imported lazily so that merely importing the registry does not pull in
    # torch — the API server registers detectors it never builds.
    from roadwatch.detection.yolo import YoloDetector

    return YoloDetector(config)


register("yolo", _build_yolo)
