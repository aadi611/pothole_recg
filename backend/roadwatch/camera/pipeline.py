"""Threaded capture with a bounded buffer (FR-CAM-02, FR-CAM-05, FR-CAM-06).

The central rule: **drop frames, never block capture, never queue
unboundedly** (NFR-REL-03). Frames are recoverable — the next one is 33 ms
away. Blocking the capture thread is not: it stalls the camera, backs up the
driver, and eventually exhausts memory.

Two buffers, deliberately separate:

- The **inference queue** is small (default 2). If detection falls behind, old
  frames are dropped, because detecting a pothole from three seconds ago is
  worse than useless — the vehicle has already hit it.
- The **rolling evidence buffer** holds recent full-resolution frames so
  capture can retrospectively pick the best one (FR-CAM-05, FR-EVD-02).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING

from roadwatch.core.logging import get_logger

if TYPE_CHECKING:
    from roadwatch.camera.base import Frame, FrameSource

log = get_logger(__name__)

# Consecutive failed reads from a *live* source before the capture thread
# gives up. A file signals exhaustion explicitly and never reaches this.
_MAX_READ_FAILURES = 100


@dataclass
class SessionMetrics:
    """Per-session counters (FR-CAM-06). Exposed via API and shown in the UI."""

    frames_read: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    read_failures: int = 0
    started_at: float = field(default_factory=time.monotonic)
    _inference_ms: deque[float] = field(default_factory=lambda: deque(maxlen=120))

    def record_inference(self, elapsed_ms: float) -> None:
        self._inference_ms.append(elapsed_ms)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def avg_fps(self) -> float:
        """Effective processed frames per second — the number that matters.

        Deliberately based on frames *processed*, not frames read: a pipeline
        reading 30 fps while detecting 4 is running at 4 fps for every purpose
        the driver cares about.
        """
        elapsed = self.elapsed_s
        return self.frames_processed / elapsed if elapsed > 0 else 0.0

    @property
    def avg_inference_ms(self) -> float:
        return sum(self._inference_ms) / len(self._inference_ms) if self._inference_ms else 0.0

    @property
    def drop_rate(self) -> float:
        return self.frames_dropped / self.frames_read if self.frames_read else 0.0

    def snapshot(self) -> dict[str, float | int]:
        return {
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "read_failures": self.read_failures,
            "elapsed_s": round(self.elapsed_s, 1),
            "avg_fps": round(self.avg_fps, 2),
            "avg_inference_ms": round(self.avg_inference_ms, 1),
            "drop_rate": round(self.drop_rate, 3),
        }


class FrameBuffer:
    """Rolling buffer of recent frames for retrospective evidence capture.

    When a track reaches VERY_NEAR the best evidence frame is usually slightly
    in the past — the defect was larger and better lit a moment before it left
    the bottom of the frame. This keeps those frames available (FR-EVD-02).

    Bounded by construction: appending past ``maxlen`` discards the oldest, so
    it cannot grow without limit during a two-hour drive (NFR-REL-07).
    """

    def __init__(self, maxlen: int = 30) -> None:
        self._frames: deque[Frame] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, frame: Frame) -> None:
        with self._lock:
            self._frames.append(frame)

    def recent(self, count: int | None = None) -> list[Frame]:
        """Most recent frames, oldest first."""
        with self._lock:
            frames = list(self._frames)
        return frames[-count:] if count else frames

    def window(self, start_frame_id: int, end_frame_id: int) -> list[Frame]:
        """Buffered frames whose ids fall within an inclusive range."""
        with self._lock:
            return [f for f in self._frames if start_frame_id <= f.frame_id <= end_frame_id]

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)


class CaptureThread:
    """Reads a FrameSource in a dedicated thread.

    The consumer calls ``next_frame()``; the producer never waits on it. If the
    consumer is slow the queue fills and the producer discards the oldest
    frame, keeping latency bounded rather than letting it grow.
    """

    def __init__(
        self,
        source: FrameSource,
        *,
        queue_size: int = 2,
        buffer_size: int = 30,
        metrics: SessionMetrics | None = None,
        drop_frames: bool = True,
    ) -> None:
        """
        Args:
            drop_frames: True for live capture — the camera cannot be paused,
                so a slow consumer must lose frames rather than stall the
                producer or grow the queue (FR-CAM-02).

                False for offline file analysis, where the producer *can* be
                paused. Dropping 94% of a recorded video means not analysing
                it, which defeats the point of the standing regression fixture
                (NFR-TST-02). A file has no real-time constraint to violate.
        """
        self._source = source
        self._drop_frames = drop_frames
        self._queue: Queue[Frame] = Queue(maxsize=queue_size)
        self.buffer = FrameBuffer(maxlen=buffer_size)
        self.metrics = metrics or SessionMetrics()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._exhausted = threading.Event()
        self._error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("capture thread already started")
        self._source.open()
        self._thread = threading.Thread(
            target=self._run, name="capture", daemon=True
        )
        self._thread.start()
        log.info("capture_thread_started", queue_size=self._queue.maxsize)

    def _run(self) -> None:
        consecutive_failures = 0
        try:
            while not self._stop.is_set():
                frame = self._source.read()

                if frame is None:
                    if self._source.is_exhausted:
                        # Normal end of a file. Not a failure, and retrying it
                        # would only spin.
                        log.info("capture_stream_ended", frames_read=self.metrics.frames_read)
                        break

                    # A live source glitched; it may recover.
                    consecutive_failures += 1
                    self.metrics.read_failures += 1
                    if consecutive_failures > _MAX_READ_FAILURES:
                        log.error(
                            "capture_giving_up",
                            consecutive_failures=consecutive_failures,
                            frames_read=self.metrics.frames_read,
                        )
                        break
                    # Yield rather than spin: the source's own reconnect
                    # backoff needs time to act.
                    time.sleep(0.01)
                    continue

                consecutive_failures = 0
                self.metrics.frames_read += 1
                self.buffer.append(frame)

                if not self._drop_frames:
                    # Offline analysis: block until the consumer has room. The
                    # timeout keeps stop() responsive.
                    while not self._stop.is_set():
                        try:
                            self._queue.put(frame, timeout=0.1)
                            break
                        except Full:
                            continue
                    continue

                try:
                    self._queue.put_nowait(frame)
                except Full:
                    # Consumer is behind. Drop the OLDEST queued frame, not the
                    # newest: the freshest view of the road is the useful one.
                    try:
                        self._queue.get_nowait()
                        self.metrics.frames_dropped += 1
                    except Empty:
                        pass
                    try:
                        self._queue.put_nowait(frame)
                    except Full:
                        self.metrics.frames_dropped += 1
        except Exception as exc:
            # Never let the capture thread die silently; the consumer must be
            # able to see why frames stopped (NFR-OBS-05).
            self._error = exc
            log.error("capture_thread_failed", error=str(exc), exc_info=True)
        finally:
            self._exhausted.set()
            log.info("capture_thread_stopped", **self.metrics.snapshot())

    def next_frame(self, timeout: float = 1.0) -> Frame | None:
        """Next frame for processing, or None when the stream is finished.

        Raises whatever killed the capture thread, so a source failure
        surfaces at the consumer rather than appearing as a silent stall.
        """
        while True:
            try:
                return self._queue.get(timeout=timeout)
            except Empty:
                if self._error is not None:
                    # `from None`: the Empty is incidental (the queue happened
                    # to be idle when we noticed). Chaining it would bury the
                    # capture thread's real traceback under a timeout.
                    raise self._error from None
                if self._exhausted.is_set():
                    return None
                if self._stop.is_set():
                    return None
                # Producer is alive but slow; keep waiting.
                continue

    def stop(self) -> None:
        """Stop capture and release the source. Idempotent (NFR-REL-05)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                log.warning("capture_thread_join_timeout")
            self._thread = None
        self._source.close()

    def __enter__(self) -> CaptureThread:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


class FrameSampler:
    """Decides which frames run inference (FR-CAM-03).

    Detection is the expensive step; tracking interpolates between detections.

    ``adaptive`` paces against the **wall clock**, which is what a live camera
    needs: it holds a target inference rate as scene complexity and thermal
    headroom vary, dropping work rather than falling behind the road.

    That same behaviour is wrong for offline file analysis. Replaying a 10 s
    video in 0.3 s leaves only a handful of wall-clock slots, so almost every
    frame is skipped and the run is not a meaningful test of detection. Use
    ``every_frame`` or ``every_nth`` for file replay, which are paced by frame
    id and therefore independent of how fast the file is read.
    """

    def __init__(
        self,
        strategy: str = "adaptive",
        *,
        every_nth: int = 2,
        target_fps: float = 15.0,
    ) -> None:
        if strategy not in ("every_frame", "every_nth", "adaptive"):
            raise ValueError(
                f"unknown sampling strategy '{strategy}'; expected "
                "every_frame, every_nth, or adaptive"
            )
        self._strategy = strategy
        self._every_nth = max(1, every_nth)
        self._target_fps = target_fps
        self._stride = 1 if strategy != "every_nth" else self._every_nth
        self._last_inference = 0.0

    def should_process(self, frame: Frame) -> bool:
        if self._strategy == "every_frame":
            return True
        if self._strategy == "every_nth":
            return frame.frame_id % self._every_nth == 0

        now = time.monotonic()
        interval = 1.0 / self._target_fps if self._target_fps > 0 else 0.0
        if now - self._last_inference >= interval:
            self._last_inference = now
            return True
        return False

    @property
    def stride(self) -> int:
        return self._stride
