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

    Implementations must not block indefinitely.

    ``read()`` returning None is deliberately ambiguous between "the stream
    ended" and "this read failed but the next may succeed" — a file at EOF and
    a camera with a dropped USB packet look identical to the consumer.
    ``is_exhausted`` disambiguates them: without it the capture thread must
    spin on retries at EOF, inflating the failure count and delaying shutdown.
    """

    def open(self) -> None:
        """Acquire the underlying device or file. Raises CameraError."""
        ...

    def read(self) -> Frame | None:
        """Return the next frame, or None if none is available right now."""
        ...

    @property
    def is_exhausted(self) -> bool:
        """True when no further frames will ever arrive.

        A finished video file is exhausted. A live camera never is — it can
        always recover — so it returns False and relies on the consumer's
        failure ceiling instead.
        """
        ...

    def close(self) -> None:
        """Release the device. Idempotent."""
        ...

    @property
    def fps(self) -> float:
        """Nominal source frame rate. 0.0 when unknown (live streams)."""
        ...
