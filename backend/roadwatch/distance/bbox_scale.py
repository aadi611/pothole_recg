"""Bounding-box scale fallback (FR-PRX-03).

Used when camera calibration is absent or the planar estimate is unusable.
Rests on a much weaker assumption than the planar method: that potholes have a
roughly typical real-world width, so apparent width in pixels indicates range.

    d ~= (real_width_m * focal_px) / apparent_width_px

That assumption is genuinely poor. Real potholes vary from 0.2 m to well over
2 m across, so a large distant pothole and a small near one can produce the
same box. **This estimator therefore always reports low confidence**, and the
state machine will warn on it but never auto-capture (FR-PRX-06) — a report
built on this number would be putting a guess into a civic complaint.

It exists because degrading to a poor estimate with honest confidence is
better than having no proximity gating at all and falling back to alerting on
mere presence, which FR-PRX-01 forbids outright.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from roadwatch.core.contracts import DistanceEstimate

if TYPE_CHECKING:
    from roadwatch.core.config import CameraGeometry
    from roadwatch.core.contracts import Detection, GpsFix

# Assumed typical pothole width. A median rather than a mean: the distribution
# has a long tail of very large defects that would skew a mean upward and make
# every estimate read as further away than it is.
_ASSUMED_WIDTH_M = 0.6

# This method never claims more than this, regardless of how clean the box
# looks. It is below the default min_distance_confidence (0.4), so capture is
# withheld by construction rather than by a caller remembering to check.
_MAX_CONFIDENCE = 0.35


class BBoxScaleDistanceEstimator:
    """Distance from apparent box width. Low confidence by design."""

    method = "bbox_scale"

    def __init__(
        self,
        camera: CameraGeometry,
        *,
        assumed_width_m: float = _ASSUMED_WIDTH_M,
    ) -> None:
        self._h_fov_deg = camera.horizontal_fov_deg
        self._assumed_width_m = assumed_width_m

    def estimate(
        self,
        det: Detection,
        frame_shape: tuple[int, int],
        gps: GpsFix | None,
    ) -> DistanceEstimate:
        _frame_h, frame_w = frame_shape
        width_px = det.bbox.width

        assumptions = [
            f"assumed pothole width {self._assumed_width_m:.2f} m",
            "apparent size used as a range proxy",
            "no camera calibration available",
        ]

        if width_px <= 1.0:
            # A degenerate box carries no scale information.
            return DistanceEstimate(
                distance_m=float("inf"),
                time_to_pothole_s=None,
                confidence=0.0,
                method=self.method,
                assumptions=[*assumptions, "bounding box too small to scale"],
            )

        focal_px = (frame_w / 2) / math.tan(math.radians(self._h_fov_deg) / 2)
        distance = (self._assumed_width_m * focal_px) / width_px

        # Confidence tapers as the box shrinks: a 10px box is a handful of
        # pixels away from being a 12px box, which is a large relative change
        # in the implied distance.
        size_factor = min(1.0, width_px / 60.0)
        confidence = round(_MAX_CONFIDENCE * size_factor, 3)

        time_to_pothole = None
        if gps is not None and gps.speed_kmh is not None and gps.speed_kmh > 0:
            time_to_pothole = distance / (gps.speed_kmh / 3.6)
            assumptions.append(f"speed {gps.speed_kmh:.0f} km/h from GPS")

        return DistanceEstimate(
            distance_m=round(distance, 2),
            time_to_pothole_s=round(time_to_pothole, 2) if time_to_pothole else None,
            confidence=confidence,
            method=self.method,
            assumptions=assumptions,
        )
