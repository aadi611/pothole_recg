"""Ultralytics YOLO implementation of the ``Detector`` protocol.

**This module and its siblings in this package are the only places in the
codebase permitted to import ``ultralytics`` or ``torch``** (FR-DET-01),
enforced by ``scripts/check_boundaries.py``.

Everything upstream sees only ``Detection`` objects, which is what makes the
model replaceable: retraining on a different dataset, or swapping to ONNX for
the edge phase, changes this file and ``config/detection.yaml`` and nothing
else.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from roadwatch.core.contracts import BBox, Detection, HazardClass
from roadwatch.core.errors import DetectorError, WeightsIntegrityError
from roadwatch.core.logging import get_logger

if TYPE_CHECKING:
    from roadwatch.core.config import DetectionConfig

log = get_logger(__name__)


def weights_fingerprint(path: Path) -> str:
    """SHA-256 of a weights file, as the model's identity (FR-DET-06).

    ``model_version`` is derived from this rather than hand-typed, so a
    detection can always be traced to the exact weights that produced it. A
    hand-written version string drifts the moment someone retrains without
    remembering to bump it.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class YoloDetector:
    """Ultralytics YOLO behind the ``Detector`` interface.

    Args:
        config: Validated detection config.
        weights_path: Overrides ``config.model.weights_path`` when supplied.
    """

    def __init__(
        self, config: DetectionConfig, *, weights_path: str | Path | None = None
    ) -> None:
        self._config = config
        self._weights = Path(weights_path or config.model.weights_path)
        self._model: Any = None
        self._warm = False

        # Map model class ids to our taxonomy, for enabled classes only. Built
        # once here rather than per-frame.
        self._class_map: dict[int, HazardClass] = {}
        for name, entry in config.classes.items():
            if entry.enabled:
                self._class_map[entry.model_class_id] = HazardClass(name)

        self.model_version = "unloaded"

    def _precision_kwargs(self) -> dict[str, Any]:
        """Half-precision argument for the installed Ultralytics version.

        Ultralytics renamed ``half`` to ``quantize`` in 8.4, and the old name
        emits a deprecation warning on *every* predict call — once per frame,
        which would drown the structured logs.

        Selected by version rather than by signature inspection: ``predict``
        takes generic ``**kwargs``, so the parameter names are not
        introspectable and probing would always silently pick the old name.
        """
        if not self._config.model.half_precision:
            return {}

        from ultralytics import __version__ as ultralytics_version

        try:
            major, minor = (int(p) for p in ultralytics_version.split(".")[:2])
        except ValueError:
            # Unparseable version (a dev build): prefer the current spelling.
            return {"quantize": "fp16"}

        if (major, minor) >= (8, 4):
            return {"quantize": "fp16"}
        return {"half": True}

    def _verify_weights(self) -> str:
        if not self._weights.is_file():
            raise DetectorError(
                f"weights not found: {self._weights}\n"
                f"Train a model first (scripts/train_detector.py) or point "
                f"config/detection.yaml at an existing .pt file."
            )

        actual = weights_fingerprint(self._weights)
        expected = self._config.model.weights_sha256

        if expected and actual != expected:
            raise WeightsIntegrityError(
                f"weights hash mismatch for {self._weights}\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}\n"
                f"The model on disk is not the one that was validated, so its "
                f"recorded mAP and tuned thresholds no longer apply."
            )
        return actual

    def warmup(self) -> None:
        """Load weights and run one dummy inference.

        The first CUDA inference costs far more than steady state. Paying that
        on the first real frame would blow the 300 ms alert budget
        (FR-ALT-06) for a pothole the driver is already approaching.
        """
        digest = self._verify_weights()
        self.model_version = f"yolo-{digest[:12]}"

        try:
            # ultralytics does not mark YOLO as an explicit re-export, so mypy
            # cannot see it; the import is valid at runtime.
            from ultralytics import YOLO  # type: ignore[attr-defined]
        except ImportError as exc:
            raise DetectorError(
                "ultralytics is not installed. Install the detection extra:\n"
                '    pip install -e ".[detection]"'
            ) from exc

        try:
            self._model = YOLO(str(self._weights))
        except Exception as exc:
            raise DetectorError(f"failed to load weights {self._weights}: {exc}") from exc

        device = self._config.model.device
        size = self._config.model.input_size
        dummy = np.zeros((size, size, 3), dtype=np.uint8)

        try:
            self._model.predict(dummy, device=device, imgsz=size, verbose=False, **self._precision_kwargs())
        except Exception as exc:
            raise DetectorError(
                f"warmup inference failed on device '{device}': {exc}\n"
                f"If CUDA is unavailable, set device: cpu in config/detection.yaml."
            ) from exc

        self._warm = True
        log.info(
            "detector_ready",
            model_version=self.model_version,
            weights=str(self._weights),
            device=device,
            input_size=size,
            classes=sorted(c.value for c in self._class_map.values()),
        )

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        """Detect hazards in one BGR frame.

        Returns only enabled classes above the configured confidence
        threshold. Empty list when nothing is found — never None.
        """
        if not self._warm:
            raise DetectorError("detect() called before warmup()")

        timestamp = datetime.now(UTC)

        try:
            results = self._model.predict(
                frame,
                device=self._config.model.device,
                imgsz=self._config.model.input_size,
                conf=self._config.thresholds.confidence,
                iou=self._config.thresholds.iou,
                max_det=self._config.thresholds.max_detections,
                verbose=False,
                **self._precision_kwargs(),
            )
        except Exception as exc:
            # One bad frame must not end a drive. Report it and continue with
            # no detections for this frame (NFR-REL-04).
            log.error("inference_failed", frame_id=frame_id, error=str(exc))
            return []

        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls.item())
                hazard = self._class_map.get(class_id)
                if hazard is None:
                    # A class the weights emit but config has disabled.
                    continue

                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        hazard_class=hazard,
                        confidence=float(box.conf.item()),
                        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        model_version=self.model_version,
                    )
                )
        return detections

    def close(self) -> None:
        """Release the model. Idempotent."""
        self._model = None
        self._warm = False
