#!/usr/bin/env python
"""Measure track-ID stability under simulated detector noise.

Phase 1's exit criterion is that one physical pothole holds one track ID
across its visible frames in >=80% of spot checks. Real detectors miss frames
and jitter boxes, so this sweeps both and reports the fraction of runs where
an approaching defect produced exactly one ID.

Usage:
    python scripts/bench_tracker.py
    python scripts/bench_tracker.py --trials 200
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from roadwatch.core.contracts import BBox, Detection, HazardClass
from roadwatch.core.logging import configure_logging
from roadwatch.tracking.tracker import Tracker

T0 = datetime(2026, 8, 18, tzinfo=UTC)


def simulate(
    *,
    miss_rate: float,
    jitter_px: float,
    iou_threshold: float,
    min_hits: int,
    max_age: int,
    frames: int,
    trials: int,
    seed: int,
) -> float:
    """Fraction of trials where the approach produced exactly one track id."""
    rng = np.random.default_rng(seed)
    clean = 0

    for _ in range(trials):
        tracker = Tracker(
            iou_threshold=iou_threshold, max_age=max_age, min_hits=min_hits
        )
        ids: set[str] = set()

        for i in range(frames):
            ts = T0 + timedelta(seconds=i / 30)

            if rng.random() < miss_rate:
                tracker.update([], ts)
                continue

            # Approaching defect: moves down the frame and grows, as on a
            # real forward-facing camera.
            y = 200 + i * 7
            size = 45 + i * 2.2
            jx, jy = rng.normal(0, jitter_px, 2)

            det = Detection(
                frame_id=i,
                timestamp=ts,
                hazard_class=HazardClass.POTHOLE,
                confidence=float(rng.uniform(0.5, 0.95)),
                bbox=BBox(x1=600 + jx, y1=y + jy, x2=600 + size + jx, y2=y + size + jy),
                model_version="bench",
            )
            ids |= {t.track_id for t in tracker.update([det], ts)}

        if len(ids) == 1:
            clean += 1

    return clean / trials


CONDITIONS = [
    ("clean", 0.00, 2.0),
    ("mild", 0.10, 4.0),
    ("realistic", 0.15, 6.0),
    ("harsh", 0.25, 8.0),
]


def simulate_separation(
    *,
    gap_px: float,
    iou_threshold: float,
    min_hits: int,
    max_age: int,
    frames: int,
    trials: int,
    seed: int,
) -> float:
    """Fraction of trials where two nearby defects stayed two tracks.

    The counterweight to loosening the IoU threshold: a looser gate bridges
    detector noise, but if it is too loose two adjacent potholes merge into
    one track, which loses a report entirely.
    """
    rng = np.random.default_rng(seed)
    correct = 0

    for _ in range(trials):
        tracker = Tracker(
            iou_threshold=iou_threshold, max_age=max_age, min_hits=min_hits
        )
        ids: set[str] = set()

        for i in range(frames):
            ts = T0 + timedelta(seconds=i / 30)
            y = 200 + i * 7
            size = 45 + i * 2.2
            dets = []
            for lane_x in (600.0, 600.0 + gap_px):
                jx, jy = rng.normal(0, 4.0, 2)
                dets.append(
                    Detection(
                        frame_id=i,
                        timestamp=ts,
                        hazard_class=HazardClass.POTHOLE,
                        confidence=float(rng.uniform(0.5, 0.95)),
                        bbox=BBox(
                            x1=lane_x + jx,
                            y1=y + jy,
                            x2=lane_x + size + jx,
                            y2=y + size + jy,
                        ),
                        model_version="bench",
                    )
                )
            ids |= {t.track_id for t in tracker.update(dets, ts)}

        if len(ids) == 2:
            correct += 1

    return correct / trials


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Quiet the per-transition logs; we want the summary only.
    configure_logging("ERROR", json_output=False)

    configs = [
        ("iou=0.30 max_age=10 (defaults)", 0.30, 10, 3),
        ("iou=0.20 max_age=10", 0.20, 10, 3),
        ("iou=0.20 max_age=15", 0.20, 15, 3),
        ("iou=0.15 max_age=15", 0.15, 15, 3),
    ]

    print(f"Single-track-ID rate over {args.trials} trials of {args.frames} frames")
    print("(Phase 1 exit criterion: >=80% under realistic noise)\n")

    header = f"{'config':<32}" + "".join(f"{name:>12}" for name, _, _ in CONDITIONS)
    print(header)
    print("-" * len(header))

    for label, iou_t, max_age, min_hits in configs:
        row = f"{label:<32}"
        for _name, miss, jitter in CONDITIONS:
            score = simulate(
                miss_rate=miss,
                jitter_px=jitter,
                iou_threshold=iou_t,
                min_hits=min_hits,
                max_age=max_age,
                frames=args.frames,
                trials=args.trials,
                seed=args.seed,
            )
            row += f"{score:>11.0%} "
        print(row)

    # The counterweight: a looser gate must not merge distinct defects.
    gaps = [60.0, 90.0, 130.0]
    print(f"\nSeparation: two distinct potholes staying two tracks ({args.trials} trials)")
    header2 = f"{'config':<32}" + "".join(f"{f'{g:.0f}px apart':>14}" for g in gaps)
    print(header2)
    print("-" * len(header2))

    for label, iou_t, max_age, min_hits in configs:
        row = f"{label:<32}"
        for gap in gaps:
            score = simulate_separation(
                gap_px=gap,
                iou_threshold=iou_t,
                min_hits=min_hits,
                max_age=max_age,
                frames=args.frames,
                trials=args.trials,
                seed=args.seed,
            )
            row += f"{score:>13.0%} "
        print(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
