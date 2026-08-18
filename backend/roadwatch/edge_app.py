"""Edge runtime — the vehicle-side pipeline (Phases 1-2).

Wires camera → detection → tracking → distance → state machine → alerts, and
renders an annotated view.

Evidence capture, GPS, and the local report queue arrive in Phase 3. Until GPS
exists, ``speed_kmh`` is None everywhere, which means the speed-suppression
rule (FR-ALT-05) cannot yet silence alerts when the vehicle is stationary.

Usage:
    python -m roadwatch.edge_app --source tests/fixtures/drive.mp4 --display
    python -m roadwatch.edge_app --source 0 --weights models/pothole.pt
    python -m roadwatch.edge_app --source drive.mp4 --no-alerts
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING

import cv2

from roadwatch.alerts.engine import AlertEngine
from roadwatch.alerts.sinks import build_sinks
from roadwatch.camera.pipeline import CaptureThread, FrameSampler, SessionMetrics
from roadwatch.camera.sources import VideoFileSource, WebcamSource
from roadwatch.core.config import load_alerts, load_detection, load_proximity
from roadwatch.core.contracts import TrackState
from roadwatch.core.errors import RoadWatchError
from roadwatch.core.logging import bind_session, configure_logging, get_logger
from roadwatch.detection import registry
from roadwatch.distance.bbox_scale import BBoxScaleDistanceEstimator
from roadwatch.distance.planar import PlanarDistanceEstimator
from roadwatch.distance.smoothing import DistanceSmoother
from roadwatch.state.machine import StateMachine, StateThresholds
from roadwatch.tracking.tracker import Tracker

if TYPE_CHECKING:
    from roadwatch.camera.base import FrameSource
    from roadwatch.tracking.tracker import TrackedObject

log = get_logger(__name__)

# Overlay colours (BGR), by track state.
_STATE_COLOURS: dict[str, tuple[int, int, int]] = {
    "UNKNOWN": (128, 128, 128),
    "DETECTED": (0, 200, 255),
    "APPROACHING": (0, 140, 255),
    "VERY_NEAR": (0, 0, 255),
    "CAPTURED": (255, 0, 255),
    "REPORTED": (0, 255, 0),
    "LOST": (80, 80, 80),
}


class _Shutdown:
    """Cooperative shutdown on SIGINT/SIGTERM (NFR-REL-05)."""

    def __init__(self) -> None:
        self.requested = False
        signal.signal(signal.SIGINT, self._handle)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        log.info("shutdown_requested", signal=signum)
        self.requested = True


def _build_source(spec: str, *, realtime: bool) -> FrameSource:
    """A digit means a camera index; anything else is a path."""
    if spec.isdigit():
        return WebcamSource(int(spec))
    return VideoFileSource(spec, realtime=realtime)


def _draw(frame_image: cv2.typing.MatLike, tracks: list[TrackedObject], metrics: SessionMetrics) -> None:
    """Annotate a frame in place with boxes, labels, and session stats."""
    for track in tracks:
        box = track.latest.bbox
        colour = _STATE_COLOURS.get(track.state.value, (255, 255, 255))
        p1 = (int(box.x1), int(box.y1))
        p2 = (int(box.x2), int(box.y2))
        cv2.rectangle(frame_image, p1, p2, colour, 2)

        # "~11 m", never "11.34 m": the estimate does not justify decimals,
        # and false precision invites misplaced trust (FR-UI-02).
        distance_label = ""
        if track.distance is not None and track.distance.distance_m != float("inf"):
            distance_label = f" ~{track.distance.distance_m:.0f}m"
            if track.distance.confidence < 0.4:
                distance_label += "?"

        label = (
            f"{track.hazard_class.value} {track.latest.confidence:.2f}"
            f"{distance_label} [{track.state.value}] {track.track_id[:6]}"
        )
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_image, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), colour, -1)
        cv2.putText(
            frame_image, label, (p1[0] + 2, p1[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    stats = (
        f"FPS {metrics.avg_fps:5.1f} | inf {metrics.avg_inference_ms:5.1f}ms | "
        f"read {metrics.frames_read} proc {metrics.frames_processed} "
        f"drop {metrics.frames_dropped}"
    )
    cv2.rectangle(frame_image, (0, 0), (frame_image.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(
        frame_image, stats, (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )


def run(args: argparse.Namespace) -> int:
    configure_logging(args.log_level, json_output=not args.human_logs)

    session_id = f"sess-{int(time.time())}"
    bind_session(session_id=session_id)

    detection_cfg = load_detection()
    proximity_cfg = load_proximity()

    if args.weights:
        detection_cfg = detection_cfg.model_copy(
            update={"model": detection_cfg.model.model_copy(update={"weights_path": args.weights})}
        )
    if args.device:
        detection_cfg = detection_cfg.model_copy(
            update={"model": detection_cfg.model.model_copy(update={"device": args.device})}
        )

    detector = registry.build(detection_cfg)
    detector.warmup()

    tracker = Tracker(
        iou_threshold=args.iou_match,
        max_age=args.max_age,
        min_hits=args.min_hits,
    )

    machine = StateMachine(
        StateThresholds(
            warn=proximity_cfg.thresholds_m.warn,
            capture=proximity_cfg.thresholds_m.capture,
            min_distance_confidence=proximity_cfg.confidence.min_distance_confidence,
        )
    )

    # Adaptive sampling paces against the wall clock, which is right for a
    # live camera but skips almost everything when a file is replayed faster
    # than realtime. Default file replay to every frame so an offline run is a
    # real test of detection.
    strategy = args.sampling or (
        detection_cfg.sampling.strategy
        if (args.source.isdigit() or args.realtime)
        else "every_frame"
    )
    sampler = FrameSampler(
        strategy,
        every_nth=detection_cfg.sampling.every_nth,
        target_fps=detection_cfg.sampling.target_inference_fps,
    )
    log.info("sampling_strategy", strategy=strategy)

    # Distance estimation (FR-PRX-02/03). The planar estimator needs measured
    # camera geometry; bbox_scale is the degraded fallback when calibration is
    # absent, and caps its own confidence below the capture threshold.
    estimator: PlanarDistanceEstimator | BBoxScaleDistanceEstimator
    if args.estimator == "bbox_scale" or proximity_cfg.estimator == "bbox_scale":
        estimator = BBoxScaleDistanceEstimator(proximity_cfg.camera)
    else:
        estimator = PlanarDistanceEstimator(proximity_cfg.camera)
    smoother = DistanceSmoother(window=proximity_cfg.smoothing.window_frames)
    log.info("distance_estimator", method=estimator.method)

    alerts_cfg = load_alerts()
    alert_engine = AlertEngine(
        alerts_cfg,
        build_sinks(alerts_cfg.sinks, rate_wpm=alerts_cfg.tts.rate_wpm)
        if not args.no_alerts
        else [],
    )

    metrics = SessionMetrics()
    shutdown = _Shutdown()
    source = _build_source(args.source, realtime=args.realtime)

    log.info(
        "session_started",
        session_id=session_id,
        source=args.source,
        model_version=detector.model_version,
    )

    exit_code = 0
    try:
        # A live camera cannot be paused, so it drops under load. A file can,
        # and analysing every frame is the point of an offline run.
        is_live = args.source.isdigit() or args.realtime
        with CaptureThread(
            source,
            queue_size=args.queue_size,
            buffer_size=args.buffer_size,
            metrics=metrics,
            drop_frames=is_live,
        ) as capture:
            while not shutdown.requested:
                frame = capture.next_frame(timeout=2.0)
                if frame is None:
                    break

                tracks = tracker.active

                if sampler.should_process(frame):
                    started = time.perf_counter()
                    detections = detector.detect(frame.image, frame.frame_id)
                    metrics.record_inference((time.perf_counter() - started) * 1000)
                    metrics.frames_processed += 1

                    tracks = tracker.update(detections, frame.timestamp)

                    for track in tracks:
                        # Estimate, then smooth per track, so the readout does
                        # not jitter and thresholds do not flap (FR-PRX-07).
                        raw = estimator.estimate(
                            track.latest, frame.shape, gps=None
                        )
                        track.distance = smoother.update(track.track_id, raw)

                        transition = machine.advance(
                            track_id=track.track_id,
                            current=track.state,
                            is_confirmed=track.hits >= args.min_hits,
                            distance=track.distance,
                        )
                        if transition is None:
                            continue

                        track.state = transition.to_state

                        # Alerts are driven by the state machine, which has
                        # already gated on distance - never on presence in
                        # frame (FR-PRX-01, FR-ALT-01).
                        if transition.to_state is TrackState.APPROACHING:
                            alert_engine.alert(
                                track_id=track.track_id,
                                level="warn",
                                distance=track.distance,
                                speed_kmh=None,
                            )
                        elif transition.to_state is TrackState.VERY_NEAR:
                            alert_engine.alert(
                                track_id=track.track_id,
                                level="near",
                                distance=track.distance,
                                speed_kmh=None,
                            )

                    # Bound per-track memory across a long drive (NFR-REL-07).
                    live = {t.track_id for t in tracks}
                    smoother.retain_only(live)
                    alert_engine.retain_only(live)

                if args.display:
                    annotated = frame.image.copy()
                    _draw(annotated, tracks, metrics)
                    cv2.imshow("RoadWatch", annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        log.info("shutdown_requested", signal="keypress")
                        break

                if args.max_frames and metrics.frames_read >= args.max_frames:
                    break

    except RoadWatchError as exc:
        log.error("session_failed", error=str(exc))
        print(f"error: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        detector.close()
        alert_engine.close()
        if args.display:
            cv2.destroyAllWindows()

        summary = metrics.snapshot()
        summary["alerts_issued"] = len(alert_engine.issued)
        log.info("session_summary", session_id=session_id, **summary)

        print("\n--- session summary ---")
        for key, value in summary.items():
            print(f"  {key:20} {value}")

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="roadwatch.edge_app", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", required=True,
        help="Video file path, or a camera index like 0",
    )
    parser.add_argument("--weights", help="Override the configured weights path")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], help="Override device")
    parser.add_argument("--display", action="store_true", help="Show the annotated view")
    parser.add_argument(
        "--realtime", action="store_true",
        help="Play a video file at its native frame rate rather than as fast as possible",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all)")
    parser.add_argument(
        "--estimator", choices=["planar", "bbox_scale"],
        help="Override the configured distance estimator",
    )
    parser.add_argument(
        "--no-alerts", action="store_true",
        help="Run detection and state transitions without emitting alerts",
    )
    parser.add_argument(
        "--sampling", choices=["every_frame", "every_nth", "adaptive"],
        help="Override the configured sampling strategy (file replay defaults to every_frame)",
    )
    parser.add_argument("--min-hits", type=int, default=3, help="Confirmations before DETECTED")
    parser.add_argument("--max-age", type=int, default=15, help="Frames unmatched before LOST")
    parser.add_argument(
        "--iou-match", type=float, default=0.15,
        help="Association IoU threshold (see scripts/bench_tracker.py)",
    )
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--buffer-size", type=int, default=30)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--human-logs", action="store_true",
        help="Console-readable logs instead of JSON",
    )

    args = parser.parse_args()

    if not args.source.isdigit() and not Path(args.source).exists():
        print(f"error: source not found: {args.source}", file=sys.stderr)
        return 1

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
