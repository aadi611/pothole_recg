"""Config validation tests.

The point of these is that bad config fails *loudly at startup*, not silently
at 60 km/h. A threshold typo that fell back to a built-in default would be a
safety bug (non-negotiable rule 5).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from roadwatch.core.config import (
    AlertsConfig,
    DetectionConfig,
    ProximityConfig,
    load_alerts,
    load_detection,
    load_proximity,
    load_yaml,
)
from roadwatch.core.errors import ConfigError


def test_shipped_proximity_config_is_valid() -> None:
    cfg = load_proximity()
    assert cfg.camera.mount_height_m > 0
    assert cfg.thresholds_m.detect > cfg.thresholds_m.warn > cfg.thresholds_m.capture


def test_shipped_detection_config_is_valid() -> None:
    cfg = load_detection()
    assert cfg.classes["pothole"].enabled


def test_shipped_alerts_config_is_valid() -> None:
    cfg = load_alerts()
    assert cfg.cooldown_s >= 0
    assert cfg.sinks


def test_missing_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_yaml(tmp_path / "nope.yaml", ProximityConfig)


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("key: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_yaml(bad, ProximityConfig)


def _proximity_dict() -> dict:
    return yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "config" / "proximity.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_out_of_order_thresholds_are_rejected(tmp_path: Path) -> None:
    """capture > warn would fire the camera before warning the driver."""
    data = _proximity_dict()
    data["thresholds_m"] = {"detect": 20.0, "warn": 4.0, "capture": 10.0}
    path = tmp_path / "proximity.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="detect > warn > capture"):
        load_yaml(path, ProximityConfig)


def test_fallback_identical_to_primary_is_rejected(tmp_path: Path) -> None:
    data = _proximity_dict()
    data["fallback_estimator"] = data["estimator"]
    path = tmp_path / "proximity.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="fallback_estimator must differ"):
        load_yaml(path, ProximityConfig)


def test_negative_mount_height_is_rejected(tmp_path: Path) -> None:
    data = _proximity_dict()
    data["camera"]["mount_height_m"] = -1.0
    path = tmp_path / "proximity.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="mount_height_m"):
        load_yaml(path, ProximityConfig)


def _detection_dict() -> dict:
    return yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "config" / "detection.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_no_enabled_class_is_rejected(tmp_path: Path) -> None:
    data = _detection_dict()
    for entry in data["classes"].values():
        entry["enabled"] = False
    path = tmp_path / "detection.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="no hazard class is enabled"):
        load_yaml(path, DetectionConfig)


def test_half_precision_on_cpu_is_rejected(tmp_path: Path) -> None:
    data = _detection_dict()
    data["model"]["device"] = "cpu"
    data["model"]["half_precision"] = True
    path = tmp_path / "detection.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="half_precision"):
        load_yaml(path, DetectionConfig)


def test_confidence_above_one_is_rejected(tmp_path: Path) -> None:
    data = _detection_dict()
    data["thresholds"]["confidence"] = 1.5
    path = tmp_path / "detection.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="confidence"):
        load_yaml(path, DetectionConfig)


def test_empty_sinks_list_is_rejected(tmp_path: Path) -> None:
    data = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "config" / "alerts.yaml").read_text(
            encoding="utf-8"
        )
    )
    data["sinks"] = []
    path = tmp_path / "alerts.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="sinks"):
        load_yaml(path, AlertsConfig)
