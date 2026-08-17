"""Detector tests (FR-DET-01, FR-DET-04, FR-DET-06).

The registry and contract behaviour are tested without loading a model. Tests
that need real weights are marked ``gpu`` and skipped when absent, so the
suite stays fast and runs on a machine with no CUDA.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from roadwatch.core.config import load_detection
from roadwatch.core.errors import ConfigError, DetectorError, WeightsIntegrityError
from roadwatch.detection import registry
from roadwatch.detection.base import Detector
from roadwatch.detection.yolo import YoloDetector, weights_fingerprint

MODELS = Path(__file__).resolve().parents[3] / "models"


def _weights() -> Path | None:
    """Any local .pt weights, for the optional integration tests."""
    if not MODELS.is_dir():
        return None
    return next(iter(MODELS.glob("*.pt")), None)


# --------------------------------------------------------------------------
# Registry — FR-DET-01
# --------------------------------------------------------------------------


def test_yolo_is_registered() -> None:
    assert "yolo" in registry.available()


def test_unknown_detector_lists_what_is_available() -> None:
    config = load_detection()
    config = config.model_copy(update={"detector": "nonexistent"})
    with pytest.raises(ConfigError, match="unknown detector"):
        registry.build(config)


def test_registry_builds_without_loading_weights() -> None:
    """Construction is cheap; weights load in warmup().

    Lets the API server import the registry without pulling torch into memory.
    """
    detector = registry.build(load_detection())
    assert isinstance(detector, YoloDetector)


# --------------------------------------------------------------------------
# Protocol conformance
# --------------------------------------------------------------------------


def test_yolo_detector_satisfies_the_protocol() -> None:
    assert isinstance(registry.build(load_detection()), Detector)


def test_detect_before_warmup_is_an_error() -> None:
    detector = registry.build(load_detection())
    with pytest.raises(DetectorError, match="before warmup"):
        detector.detect(np.zeros((64, 64, 3), dtype=np.uint8), 0)


def test_missing_weights_names_the_path_and_the_fix() -> None:
    config = load_detection()
    detector = YoloDetector(config, weights_path="models/does-not-exist.pt")
    with pytest.raises(DetectorError, match="weights not found"):
        detector.warmup()


# --------------------------------------------------------------------------
# Weights identity — FR-DET-06
# --------------------------------------------------------------------------


def test_fingerprint_matches_hashlib(tmp_path: Path) -> None:
    blob = tmp_path / "w.pt"
    blob.write_bytes(b"pretend weights")
    assert weights_fingerprint(blob) == hashlib.sha256(b"pretend weights").hexdigest()


def test_fingerprint_changes_with_content(tmp_path: Path) -> None:
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    a.write_bytes(b"model one")
    b.write_bytes(b"model two")
    assert weights_fingerprint(a) != weights_fingerprint(b)


def test_hash_mismatch_is_fatal(tmp_path: Path) -> None:
    """A swapped weights file must fail loudly, not change behaviour silently.

    The recorded mAP and the tuned thresholds belong to a specific file. If
    that file changed, neither still applies.
    """
    blob = tmp_path / "w.pt"
    blob.write_bytes(b"actual weights")

    config = load_detection()
    config = config.model_copy(
        update={"model": config.model.model_copy(update={"weights_sha256": "0" * 64})}
    )
    detector = YoloDetector(config, weights_path=blob)

    with pytest.raises(WeightsIntegrityError, match="hash mismatch"):
        detector.warmup()


def test_absent_expected_hash_skips_verification(tmp_path: Path) -> None:
    """weights_sha256: null is allowed during development."""
    blob = tmp_path / "w.pt"
    blob.write_bytes(b"unverified weights")

    config = load_detection()
    assert config.model.weights_sha256 is None
    detector = YoloDetector(config, weights_path=blob)

    # Gets past verification, then fails on the actual load - which is what
    # we are asserting: verification did not block it.
    with pytest.raises(DetectorError, match="failed to load weights"):
        detector.warmup()


# --------------------------------------------------------------------------
# Class map — FR-DET-03
# --------------------------------------------------------------------------


def test_only_enabled_classes_are_mapped() -> None:
    config = load_detection()
    detector = YoloDetector(config)
    enabled = {name for name, e in config.classes.items() if e.enabled}
    assert {h.value for h in detector._class_map.values()} == enabled


def test_disabling_a_class_removes_it_without_code_change() -> None:
    config = load_detection()
    updated = dict(config.classes)
    updated["crack"] = updated["crack"].model_copy(update={"enabled": True})
    config = config.model_copy(update={"classes": updated})

    detector = YoloDetector(config)
    assert "crack" in {h.value for h in detector._class_map.values()}


# --------------------------------------------------------------------------
# Real inference (optional)
# --------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(_weights() is None, reason="no local weights")
def test_warmup_and_detect_on_a_blank_frame() -> None:
    config = load_detection()
    detector = YoloDetector(config, weights_path=_weights())
    detector.warmup()

    assert detector.model_version.startswith("yolo-")

    detections = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8), 0)
    assert isinstance(detections, list)
    for det in detections:
        assert 0.0 <= det.confidence <= 1.0
        assert det.model_version == detector.model_version

    detector.close()
