"""Camera pipeline tests (FR-CAM-02, FR-CAM-05, FR-CAM-06).

The behaviour under test is the one that matters under load: when inference
falls behind, frames are dropped and capture keeps running. A pipeline that
blocks instead would stall the camera and grow memory without bound.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import numpy as np
import pytest

from roadwatch.camera.base import Frame
from roadwatch.camera.pipeline import (
    CaptureThread,
    FrameBuffer,
    FrameSampler,
    SessionMetrics,
)
from roadwatch.camera.sources import SyntheticSource, VideoFileSource
from roadwatch.core.errors import CameraError


def _frame(frame_id: int) -> Frame:
    return Frame(
        frame_id=frame_id,
        timestamp=datetime.now(UTC),
        image=np.zeros((4, 4, 3), dtype=np.uint8),
    )


# --------------------------------------------------------------------------
# FrameBuffer — FR-CAM-05
# --------------------------------------------------------------------------


def test_buffer_is_bounded() -> None:
    """Must not grow over a two-hour drive (NFR-REL-07)."""
    buf = FrameBuffer(maxlen=10)
    for i in range(1000):
        buf.append(_frame(i))
    assert len(buf) == 10


def test_buffer_keeps_the_newest_frames() -> None:
    buf = FrameBuffer(maxlen=5)
    for i in range(20):
        buf.append(_frame(i))
    ids = [f.frame_id for f in buf.recent()]
    assert ids == [15, 16, 17, 18, 19]


def test_buffer_window_selects_by_frame_id() -> None:
    """Evidence capture asks for the frames around a VERY_NEAR transition."""
    buf = FrameBuffer(maxlen=50)
    for i in range(50):
        buf.append(_frame(i))
    window = buf.window(10, 14)
    assert [f.frame_id for f in window] == [10, 11, 12, 13, 14]


def test_buffer_window_outside_range_is_empty() -> None:
    buf = FrameBuffer(maxlen=5)
    for i in range(5):
        buf.append(_frame(i))
    assert buf.window(100, 200) == []


# --------------------------------------------------------------------------
# SessionMetrics — FR-CAM-06
# --------------------------------------------------------------------------


def test_avg_fps_counts_processed_not_read() -> None:
    """A pipeline reading 30fps but detecting 4 is running at 4fps."""
    m = SessionMetrics()
    m.frames_read = 300
    m.frames_processed = 40
    time.sleep(0.05)
    assert m.avg_fps < m.frames_read / m.elapsed_s


def test_drop_rate() -> None:
    m = SessionMetrics()
    m.frames_read = 100
    m.frames_dropped = 25
    assert m.drop_rate == 0.25


def test_drop_rate_with_no_frames_does_not_divide_by_zero() -> None:
    assert SessionMetrics().drop_rate == 0.0


def test_inference_window_is_bounded() -> None:
    m = SessionMetrics()
    for _ in range(1000):
        m.record_inference(10.0)
    assert m.avg_inference_ms == 10.0


def test_snapshot_has_the_ui_fields() -> None:
    snap = SessionMetrics().snapshot()
    for key in ("frames_read", "frames_processed", "frames_dropped", "avg_fps", "avg_inference_ms"):
        assert key in snap


# --------------------------------------------------------------------------
# CaptureThread — FR-CAM-02
# --------------------------------------------------------------------------


def test_reads_every_frame_when_consumer_keeps_up() -> None:
    with CaptureThread(SyntheticSource(frames=20), queue_size=4) as cap:
        seen = []
        while (frame := cap.next_frame(timeout=2.0)) is not None:
            seen.append(frame.frame_id)
    assert seen == list(range(20))
    assert cap.metrics.frames_dropped == 0


def test_drops_frames_rather_than_blocking_capture() -> None:
    """The core FR-CAM-02 guarantee.

    A slow consumer must not stall the producer. Frames are dropped, capture
    continues, and the drop is counted rather than hidden.
    """
    cap = CaptureThread(SyntheticSource(frames=200), queue_size=2)
    cap.start()

    # Consume far slower than the producer generates.
    consumed = 0
    for _ in range(5):
        if cap.next_frame(timeout=2.0) is not None:
            consumed += 1
        time.sleep(0.03)
    cap.stop()

    assert cap.metrics.frames_read > consumed
    assert cap.metrics.frames_dropped > 0, "expected drops under a slow consumer"


def test_queue_never_exceeds_its_bound() -> None:
    cap = CaptureThread(SyntheticSource(frames=500), queue_size=3)
    cap.start()
    time.sleep(0.2)
    assert cap._queue.qsize() <= 3
    cap.stop()


def test_next_frame_returns_none_at_end_of_stream() -> None:
    with CaptureThread(SyntheticSource(frames=3), queue_size=4) as cap:
        for _ in range(3):
            assert cap.next_frame(timeout=2.0) is not None
        assert cap.next_frame(timeout=2.0) is None


def test_stop_is_idempotent() -> None:
    cap = CaptureThread(SyntheticSource(frames=5))
    cap.start()
    cap.stop()
    cap.stop()


def test_eof_is_not_counted_as_a_read_failure() -> None:
    """A file ending is normal, not an error.

    Regression: read() returning None conflated EOF with a transient glitch,
    so finishing a video logged 100+ phantom failures and spun for seconds
    before stopping.
    """
    with CaptureThread(SyntheticSource(frames=10), queue_size=4) as cap:
        while cap.next_frame(timeout=2.0) is not None:
            pass
    assert cap.metrics.read_failures == 0
    assert cap.metrics.frames_read == 10


def test_exhausted_source_stops_the_producer_promptly() -> None:
    """The capture thread must exit at EOF, not spin on retries.

    Timed on the producer finishing rather than on next_frame() returning,
    since the final next_frame() legitimately waits out its own timeout on a
    drained queue.
    """
    cap = CaptureThread(SyntheticSource(frames=5), queue_size=16)
    started = time.monotonic()
    cap.start()
    cap._exhausted.wait(timeout=3.0)
    elapsed = time.monotonic() - started
    cap.stop()

    assert cap._exhausted.is_set(), "producer did not finish"
    assert elapsed < 1.0, f"producer took {elapsed:.2f}s to notice EOF"
    assert cap.metrics.read_failures == 0


def test_lossless_mode_drops_nothing_for_a_slow_consumer() -> None:
    """Offline file analysis must see every frame (NFR-TST-02).

    Dropping 94% of the regression fixture would mean not testing against it.
    """
    cap = CaptureThread(SyntheticSource(frames=40), queue_size=2, drop_frames=False)
    cap.start()
    seen = []
    while (frame := cap.next_frame(timeout=3.0)) is not None:
        seen.append(frame.frame_id)
        time.sleep(0.002)
    cap.stop()

    assert seen == list(range(40))
    assert cap.metrics.frames_dropped == 0


def test_rolling_buffer_populated_during_capture() -> None:
    with CaptureThread(SyntheticSource(frames=50), queue_size=2, buffer_size=10) as cap:
        while cap.next_frame(timeout=2.0) is not None:
            pass
        assert len(cap.buffer) == 10


# --------------------------------------------------------------------------
# FrameSampler — FR-CAM-03
# --------------------------------------------------------------------------


def test_every_frame_strategy_processes_all() -> None:
    s = FrameSampler("every_frame")
    assert all(s.should_process(_frame(i)) for i in range(10))


def test_every_nth_strategy() -> None:
    s = FrameSampler("every_nth", every_nth=3)
    assert [i for i in range(12) if s.should_process(_frame(i))] == [0, 3, 6, 9]


def test_adaptive_strategy_throttles_to_target_rate() -> None:
    s = FrameSampler("adaptive", target_fps=50.0)
    processed = sum(1 for i in range(200) if s.should_process(_frame(i)))
    # 200 frames arriving instantly must not all run inference.
    assert processed < 200


def test_frame_id_strategies_are_independent_of_read_speed() -> None:
    """every_nth must not depend on how fast the file is read.

    Adaptive paces on the wall clock, which is right for a live camera but
    means a video replayed faster than realtime gets almost no inference.
    """
    s = FrameSampler("every_nth", every_nth=2)
    assert sum(1 for i in range(100) if s.should_process(_frame(i))) == 50


def test_unknown_sampling_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown sampling strategy"):
        FrameSampler("sometimes")


# --------------------------------------------------------------------------
# Sources — FR-CAM-04
# --------------------------------------------------------------------------


def test_missing_video_file_raises_with_the_path() -> None:
    src = VideoFileSource("does/not/exist.mp4")
    with pytest.raises(CameraError, match="not found"):
        src.open()


def test_read_before_open_is_an_error() -> None:
    with pytest.raises(CameraError, match="before open"):
        VideoFileSource("x.mp4").read()


def test_synthetic_source_is_deterministic() -> None:
    a = SyntheticSource(frames=3, seed=42)
    b = SyntheticSource(frames=3, seed=42)
    a.open()
    b.open()
    fa, fb = a.read(), b.read()
    assert fa is not None and fb is not None
    assert np.array_equal(fa.image, fb.image)


def test_frame_shape_is_height_width() -> None:
    f = Frame(
        frame_id=0,
        timestamp=datetime.now(UTC),
        image=np.zeros((480, 640, 3), dtype=np.uint8),
    )
    assert f.shape == (480, 640)
