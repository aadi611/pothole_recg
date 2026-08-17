"""Configuration loading and validation.

Every tunable in this system lives in ``config/*.yaml`` and is validated here
by a Pydantic model at startup (non-negotiable rule 5: no hard-coded
thresholds). Invalid config is a hard failure with a readable message, never a
silent default — a mistyped ``warn`` threshold that quietly fell back to a
built-in value would be a safety bug, not a config bug.

Secrets never appear in YAML. They come from the environment via
``Settings`` (NFR-SEC-04).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from roadwatch.core.errors import ConfigError

# ---------------------------------------------------------------------------
# Proximity
# ---------------------------------------------------------------------------


class CameraGeometry(BaseModel):
    """Physical mounting of the camera.

    These values are the basis of every planar distance estimate. They must be
    measured on the actual vehicle, not guessed — see docs/ASSUMPTIONS.md.
    """

    id: str
    mount_height_m: float = Field(gt=0, le=5)
    pitch_deg: float = Field(ge=-90, le=90)
    horizontal_fov_deg: float = Field(gt=0, lt=180)
    resolution: tuple[int, int]
    calibration_file: str | None = None


class ProximityThresholds(BaseModel):
    detect: float = Field(gt=0)
    warn: float = Field(gt=0)
    capture: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> ProximityThresholds:
        if not self.detect > self.warn > self.capture:
            raise ValueError(
                f"thresholds must satisfy detect > warn > capture, got "
                f"detect={self.detect}, warn={self.warn}, capture={self.capture}. "
                "A capture threshold above the warn threshold would fire the "
                "camera before the driver was warned."
            )
        return self


class TimeToPothole(BaseModel):
    warn: float = Field(gt=0)


class DistanceConfidence(BaseModel):
    min_distance_confidence: float = Field(ge=0, le=1)


class Smoothing(BaseModel):
    window_frames: int = Field(ge=1, le=60)


class ProximityConfig(BaseModel):
    camera: CameraGeometry
    estimator: Literal["planar", "bbox_scale", "monodepth", "stereo"]
    fallback_estimator: Literal["planar", "bbox_scale", "monodepth", "stereo"]
    thresholds_m: ProximityThresholds
    time_to_pothole_s: TimeToPothole
    confidence: DistanceConfidence
    smoothing: Smoothing

    @model_validator(mode="after")
    def _fallback_differs(self) -> ProximityConfig:
        if self.estimator == self.fallback_estimator:
            raise ValueError(
                f"fallback_estimator must differ from estimator "
                f"(both are '{self.estimator}'); a fallback identical to the "
                "primary provides no degradation path (NFR-REL-04)."
            )
        return self


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    weights_path: str
    weights_sha256: str | None = None
    input_size: int = Field(ge=32, le=2048)
    device: Literal["cpu", "cuda", "mps"]
    half_precision: bool = False

    @model_validator(mode="after")
    def _half_precision_needs_gpu(self) -> ModelConfig:
        if self.half_precision and self.device == "cpu":
            raise ValueError(
                "half_precision is not supported on device 'cpu'; "
                "set half_precision: false or use a CUDA device."
            )
        return self


class DetectionThresholds(BaseModel):
    confidence: float = Field(ge=0, le=1)
    iou: float = Field(ge=0, le=1)
    max_detections: int = Field(ge=1, le=1000)


class ClassEntry(BaseModel):
    enabled: bool
    model_class_id: int = Field(ge=0)


class SamplingConfig(BaseModel):
    strategy: Literal["every_frame", "every_nth", "adaptive"]
    every_nth: int = Field(ge=1)
    target_inference_fps: float = Field(gt=0)


class DetectionConfig(BaseModel):
    detector: str
    model: ModelConfig
    thresholds: DetectionThresholds
    classes: dict[str, ClassEntry]
    sampling: SamplingConfig

    @model_validator(mode="after")
    def _at_least_one_class(self) -> DetectionConfig:
        if not any(c.enabled for c in self.classes.values()):
            raise ValueError(
                "no hazard class is enabled; the detector would emit nothing."
            )
        return self


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


class TtsConfig(BaseModel):
    engine: str
    warn_template: str
    near_template: str
    rate_wpm: int = Field(ge=80, le=400)


class AlertsConfig(BaseModel):
    cooldown_s: float = Field(ge=0)
    global_min_gap_s: float = Field(ge=0)
    sinks: list[str] = Field(min_length=1)
    tts: TtsConfig
    suppress_below_speed_kmh: float = Field(ge=0)
    latency_budget_ms: int = Field(gt=0)


# ---------------------------------------------------------------------------
# Secrets (environment only — never YAML)
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Secrets and deployment-specific values, from the environment.

    Nothing here may be committed. ``.env.example`` documents each var with a
    dummy value (NFR-SEC-04).
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="ROADWATCH_", extra="ignore"
    )

    database_url: str = "postgresql+psycopg://roadwatch:roadwatch@localhost:5432/roadwatch"
    jwt_secret: str = "dev-only-change-me"
    jwt_access_ttl_minutes: int = 30
    jwt_refresh_ttl_days: int = 30
    storage_root: Path = Path("storage")
    nominatim_url: str = "https://nominatim.openstreetmap.org"
    nominatim_user_agent: str = "roadwatch/0.1 (contact: set-me@example.com)"
    log_level: str = "INFO"
    app_version: str = "0.1.0"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_T = TypeVar("_T", bound=BaseModel)

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def load_yaml(path: Path, model: type[_T]) -> _T:
    """Load and validate one YAML config file.

    Raises:
        ConfigError: if the file is missing, unparseable, or fails validation.
            The message names the file and the offending field — a config
            error at startup should tell you exactly what to fix.
    """
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")

    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        details = "\n".join(
            f"    {'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        raise ConfigError(f"{path}: invalid configuration:\n{details}") from exc


def load_proximity(config_dir: Path | None = None) -> ProximityConfig:
    return load_yaml((config_dir or CONFIG_DIR) / "proximity.yaml", ProximityConfig)


def load_detection(config_dir: Path | None = None) -> DetectionConfig:
    return load_yaml((config_dir or CONFIG_DIR) / "detection.yaml", DetectionConfig)


def load_alerts(config_dir: Path | None = None) -> AlertsConfig:
    return load_yaml((config_dir or CONFIG_DIR) / "alerts.yaml", AlertsConfig)
