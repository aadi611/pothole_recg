"""Concrete frame sources (FR-CAM-01, FR-CAM-04).

Every source tolerates a corrupt or zero-byte read without crashing. A single
bad frame in a two-hour drive must not end the session.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from roadwatch.camera.base import Frame
from roadwatch.core.errors import CameraError
from roadwatch.core.logging import get_logger

log = get_logger(__name__)

# Consecutive failed reads before a live source is considered disconnected.
_MAX_CONSECUTIVE_FAILURES = 5


class VideoFileSource:
    """Frames from a recorded video.

    The mandatory source: the standing regression fixture
    (``tests/fixtures/drive.mp4``) is played through this, so every phase can
    be verified without a camera or a vehicle (NFR-TST-02).

    Timestamps are derived from the video's own presentation time, not the
    wall clock, so a replay is reproducible and can be synchronised against a
    GPX track (FR-GEO-01).
    """

    def __init__(self, path: str | Path, *, realtime: bool = False) -> None:
        """
        Args:
            path: Video file.
            realtime: Sleep between frames to play at the source frame rate.
                Off by default so tests and batch analysis run at full speed.
        """
        self._path = Path(path)
        self._realtime = realtime
        self._cap: cv2.VideoCapture | None = None
        self._frame_id = 0
        self._fps = 0.0
        self._start_wall = 0.0
        self._epoch = datetime.now(UTC)
        self._exhausted = False

    def open(self) -> None:
        if not self._path.is_file():
            raise CameraError(f"video file not found: {self._path}")

        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise CameraError(
                f"could not open video: {self._path}. The file may be corrupt "
                f"or use a codec OpenCV was not built with."
            )

        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._start_wall = time.monotonic()
        self._epoch = datetime.now(UTC)

        log.info(
            "video_source_opened",
            path=str(self._path),
            fps=round(self._fps, 2),
            frame_count=frames,
            duration_s=round(frames / self._fps, 1) if self._fps else None,
        )

    def read(self) -> Frame | None:
        if self._cap is None:
            raise CameraError("read() before open()")

        ok, image = self._cap.read()
        if not ok or image is None or image.size == 0:
            # For a file this means end of stream, which is normal. Mark it so
            # the consumer stops immediately instead of retrying a file that
            # will never produce another frame.
            self._exhausted = True
            return None

        # Presentation timestamp from the container, so replay is
        # deterministic and re-runnable.
        pos_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        if pos_ms <= 0 and self._fps > 0:
            pos_ms = 1000.0 * self._frame_id / self._fps

        if self._realtime and self._fps > 0:
            target = self._start_wall + (pos_ms / 1000.0)
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)

        frame = Frame(
            frame_id=self._frame_id,
            timestamp=datetime.fromtimestamp(
                self._epoch.timestamp() + pos_ms / 1000.0, tz=UTC
            ),
            image=image,
        )
        self._frame_id += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("video_source_closed", frames_read=self._frame_id)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_exhausted(self) -> bool:
        return self._exhausted


class WebcamSource:
    """Frames from a USB/integrated camera or an RTSP stream.

    Unlike a file, a live source can fail transiently and recover. A handful
    of bad reads is normal (USB hiccup, dropped RTSP packet); sustained
    failure means the camera is gone and warrants a reconnect (FR-CAM-04).
    """

    def __init__(
        self,
        device: int | str = 0,
        *,
        resolution: tuple[int, int] | None = None,
        reconnect_backoff_s: float = 1.0,
        max_backoff_s: float = 30.0,
    ) -> None:
        self._device = device
        self._resolution = resolution
        self._backoff = reconnect_backoff_s
        self._initial_backoff = reconnect_backoff_s
        self._max_backoff = max_backoff_s
        self._cap: cv2.VideoCapture | None = None
        self._frame_id = 0
        self._failures = 0
        self._fps = 0.0

    def open(self) -> None:
        cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            raise CameraError(f"could not open camera device: {self._device!r}")

        if self._resolution is not None:
            w, h = self._resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            actual = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            if actual != (w, h):
                # Not fatal, but distance estimation assumes the configured
                # resolution, so a silent mismatch would skew every estimate.
                log.warning(
                    "camera_resolution_mismatch",
                    requested=[w, h],
                    actual=list(actual),
                    impact="distance estimates assume the configured resolution",
                )

        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        self._failures = 0
        self._backoff = self._initial_backoff
        log.info("camera_opened", device=str(self._device), fps=round(self._fps, 2))

    def _reconnect(self) -> None:
        log.warning("camera_reconnecting", device=str(self._device), backoff_s=self._backoff)
        self.close()
        time.sleep(self._backoff)
        self._backoff = min(self._backoff * 2, self._max_backoff)
        try:
            self.open()
        except CameraError as exc:
            # Stay in the retry loop rather than killing the session; the
            # driver may simply have knocked a cable loose.
            log.error("camera_reconnect_failed", error=str(exc))

    def read(self) -> Frame | None:
        if self._cap is None:
            raise CameraError("read() before open()")

        ok, image = self._cap.read()

        if not ok or image is None or image.size == 0:
            self._failures += 1
            log.warning(
                "camera_read_failed",
                consecutive_failures=self._failures,
                frame_id=self._frame_id,
            )
            if self._failures >= _MAX_CONSECUTIVE_FAILURES:
                self._reconnect()
                self._failures = 0
            # Signal "no frame this tick" without ending the session.
            return None

        if self._failures:
            log.info("camera_recovered", after_failures=self._failures)
            self._failures = 0
            self._backoff = self._initial_backoff

        frame = Frame(
            frame_id=self._frame_id, timestamp=datetime.now(UTC), image=image
        )
        self._frame_id += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_exhausted(self) -> bool:
        """Never. A live camera can always recover, so the consumer's
        consecutive-failure ceiling decides when to give up."""
        return False


class SyntheticSource:
    """Generated frames, for tests that need no media files.

    Deterministic: the same seed yields the same frames, so a test asserting
    on detection counts stays stable.
    """

    def __init__(self, *, frames: int = 100, width: int = 640, height: int = 480, seed: int = 0) -> None:
        self._total = frames
        self._width = width
        self._height = height
        self._rng = np.random.default_rng(seed)
        self._frame_id = 0

    def open(self) -> None:
        self._frame_id = 0

    def read(self) -> Frame | None:
        if self._frame_id >= self._total:
            return None
        image = self._rng.integers(
            0, 255, (self._height, self._width, 3), dtype=np.uint8
        )
        frame = Frame(
            frame_id=self._frame_id, timestamp=datetime.now(UTC), image=image
        )
        self._frame_id += 1
        return frame

    def close(self) -> None:
        pass

    @property
    def fps(self) -> float:
        return 30.0

    @property
    def is_exhausted(self) -> bool:
        return self._frame_id >= self._total
