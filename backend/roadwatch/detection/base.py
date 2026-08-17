"""The ``Detector`` interface (FR-DET-01).

This is the seam that keeps the model swappable. Everything upstream of
detection — tracking, distance, alerts, evidence — codes against this protocol
and never against a model library. Only implementations in this package may
import ``ultralytics`` or ``torch``; ``scripts/check_boundaries.py`` enforces
that in CI.

Concretely, this is what lets us ship on Roboflow-trained weights now and
retrain on RDD2022's India subset later as a config change rather than a
rewrite (see docs/DATASETS.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from roadwatch.core.contracts import Detection


@runtime_checkable
class Detector(Protocol):
    """A source of per-frame detections.

    Implementations must be safe to call from the inference thread and must
    not block on I/O after ``warmup()``.
    """

    model_version: str
    """Identity of the weights actually loaded.

    Derived from the weights file hash, never a hand-typed string (FR-DET-06).
    Stamped onto every Detection and every report, so a detection can always
    be traced back to the exact model that produced it.
    """

    def warmup(self) -> None:
        """Load weights and run a dummy inference.

        First inference on a CUDA device costs far more than steady-state, and
        paying that during the first real frame would blow the FR-ALT-06
        latency budget on a pothole the driver is already approaching.
        """
        ...

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        """Detect hazards in one BGR frame.

        Args:
            frame: HxWx3 BGR image, as delivered by the camera source.
            frame_id: Monotonic frame counter, stamped onto each Detection.

        Returns:
            Detections above the configured confidence threshold, for enabled
            classes only. Empty list when nothing is found — never None.
        """
        ...

    def close(self) -> None:
        """Release model resources. Idempotent."""
        ...
