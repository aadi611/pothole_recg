"""Frame source interface (FR-CAM-01).

Sources are pluggable: video file, webcam, RTSP, phone stream. The video file
source is mandatory — it is how the whole system gets tested reproducibly
without a vehicle (NFR-DEV-02).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class Frame:
    """One captured frame with its provenance.

    ``timestamp`` is capture time, not processing time. Evidence binding
    (FR-EVD-04) matches a GPS fix against this, so a processing-time stamp
    would silently mis-locate every report by the inference latency.
    """

    frame_id: int
    timestamp: datetime
    image: np.ndarray
    """HxWx3 BGR, as OpenCV delivers it."""

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) in pixels."""
        h, w = self.image.shape[:2]
        return int(h), int(w)


@runtime_checkable
class FrameSource(Protocol):
    """A source of frames.

    Implementations must not block indefinitely. ``read()`` returning None
    means the stream ended; transient failures are handled internally with
    backoff (FR-CAM-04).
    """

    def open(self) -> None:
        """Acquire the underlying device or file. Raises CameraError."""
        ...

    def read(self) -> Frame | None:
        """Return the next frame, or None when the stream is exhausted."""
        ...

    def close(self) -> None:
        """Release the device. Idempotent."""
        ...

    @property
    def fps(self) -> float:
        """Nominal source frame rate. 0.0 when unknown (live streams)."""
        ...
