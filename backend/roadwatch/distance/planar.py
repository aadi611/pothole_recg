"""Road-plane distance estimation (FR-PRX-02).

The primary estimator. Projects the bottom-centre of a bounding box onto an
assumed flat road plane using the camera's mount height, pitch, and vertical
field of view.

Geometry
--------

With the camera at height ``h`` above the road, pitched down by ``theta``, a
pixel row ``y`` corresponds to a ray leaving the camera at some angle below
horizontal. If that total depression angle is ``alpha``, the ray meets the
road at::

    d = h / tan(alpha)

The whole method is that one equation. Everything else is working out
``alpha`` for a given pixel row.

For a pinhole camera with vertical field of view ``fov_v`` and image height
``H``, the angular offset of row ``y`` from the optical axis is::

    offset = atan((y_centre - y) / f_pixels)
    f_pixels = (H / 2) / tan(fov_v / 2)

and ``alpha = theta + offset``, taking ``theta`` as positive downward.

Why the bottom edge
-------------------

``bbox.bottom_center`` is where the defect meets the road — its ground-contact
point. The top edge is somewhere above the surface and would project further
away. This is precisely why detection must emit boxes and a classifier cannot
serve this system.

Failure modes
-------------

The flat-road assumption is the weak point, and it fails in ways that matter:

- **Uphill** ahead: the road rises into the ray, so the true distance is
  *shorter* than computed. The system warns late. This is the dangerous
  direction, which is why confidence falls off with distance.
- **Downhill**: the estimate is short, so the system warns early. Annoying,
  not dangerous.
- **Pitch changes**: braking, acceleration, and speed bumps all rotate the
  camera. A 1-degree pitch error at 15 m is roughly a 2-3 m distance error.
- **Long range**: near the horizon a single pixel row spans several metres,
  so quantisation alone dominates the estimate.

See ``docs/DISTANCE_LIMITATIONS.md``. These are stated in the returned
``assumptions`` list so the UI can surface them.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from roadwatch.core.contracts import DistanceEstimate
from roadwatch.core.errors import CalibrationError

if TYPE_CHECKING:
    from roadwatch.core.config import CameraGeometry
    from roadwatch.core.contracts import Detection, GpsFix

# Below this depression angle the ray is effectively parallel to the road and
# `h / tan(alpha)` explodes toward infinity. Anything at or above the horizon
# is not a road surface point we can locate.
_MIN_DEPRESSION_RAD = math.radians(0.35)

# Distance beyond which the estimate is not useful for alerting. Reported, but
# with confidence approaching zero.
_MAX_USEFUL_M = 120.0


def vertical_fov_deg(horizontal_fov_deg: float, width: int, height: int) -> float:
    """Derive vertical FOV from horizontal FOV and aspect ratio.

    Configs specify horizontal FOV because that is what camera datasheets
    quote, but the geometry needs vertical.

    Uses the pinhole relation rather than scaling the angle by the aspect
    ratio: angles do not scale linearly, and the naive version is several
    degrees out at typical dashcam fields of view.
    """
    if width <= 0 or height <= 0:
        raise CalibrationError(f"invalid frame dimensions: {width}x{height}")
    if not 0 < horizontal_fov_deg < 180:
        raise CalibrationError(
            f"horizontal_fov_deg must be in (0, 180), got {horizontal_fov_deg}"
        )

    h_fov = math.radians(horizontal_fov_deg)
    focal_px = (width / 2) / math.tan(h_fov / 2)
    return math.degrees(2 * math.atan((height / 2) / focal_px))


def horizon_row(frame_height: int, pitch_deg: float, v_fov_deg: float) -> float:
    """Pixel row where the road plane meets the horizon.

    Anything above this row cannot be a point on the road ahead. Useful for
    rejecting detections on walls, vehicles, or sky.
    """
    focal_px = (frame_height / 2) / math.tan(math.radians(v_fov_deg) / 2)
    # Positive pitch_deg here means "pitched down"; the horizon sits above the
    # image centre by focal * tan(pitch).
    return (frame_height / 2) - focal_px * math.tan(math.radians(pitch_deg))


class PlanarDistanceEstimator:
    """Ground-plane projection using known camera geometry.

    Args:
        camera: Validated geometry from ``config/proximity.yaml``. Mount
            height and pitch must be *measured* on the vehicle — the estimate
            is only as good as they are.
    """

    method = "planar"

    def __init__(self, camera: CameraGeometry) -> None:
        # `pitch_deg` is negative-down in config (matching how people describe
        # a camera angled at the road); the maths wants positive-down.
        self._depression_deg = -camera.pitch_deg
        self._height_m = camera.mount_height_m
        self._h_fov_deg = camera.horizontal_fov_deg

        if self._depression_deg <= 0:
            raise CalibrationError(
                f"camera pitch must be angled down at the road; got "
                f"pitch_deg={camera.pitch_deg} (expected negative). A camera "
                f"level or angled up cannot see the road plane ahead."
            )

        self._assumptions = [
            "flat road surface",
            f"camera height {self._height_m:.2f} m",
            f"camera pitch {camera.pitch_deg:.1f} deg",
            "fixed camera pose (no braking/acceleration pitch change)",
        ]

    def _distance_for_row(self, y: float, frame_h: int, frame_w: int) -> float | None:
        """Ground distance for a pixel row, or None if at/above the horizon."""
        v_fov = vertical_fov_deg(self._h_fov_deg, frame_w, frame_h)
        focal_px = (frame_h / 2) / math.tan(math.radians(v_fov) / 2)

        # Rows below centre look further down, so add to the depression angle.
        offset_rad = math.atan((y - frame_h / 2) / focal_px)
        alpha = math.radians(self._depression_deg) + offset_rad

        if alpha <= _MIN_DEPRESSION_RAD:
            # At or above the horizon: no finite intersection with the road.
            return None

        return self._height_m / math.tan(alpha)

    def _confidence(
        self, distance_m: float, y: float, frame_h: int, frame_w: int
    ) -> float:
        """How much to trust this estimate.

        Falls off with distance for a concrete reason: near the horizon each
        pixel row spans a large ground distance, so a one-pixel box error
        becomes a several-metre distance error. Confidence encodes that
        sensitivity, not a vague feeling.
        """
        if distance_m <= 0:
            return 0.0

        # Sensitivity: how much distance changes for one row of pixel error.
        # Computed numerically because the analytic derivative adds nothing
        # here and this is clearer to check.
        neighbour = self._distance_for_row(y - 1.0, frame_h, frame_w)
        if neighbour is None:
            return 0.05
        per_pixel_m = abs(distance_m - neighbour)

        # Relative error per pixel. Under ~1% is good; over ~10% is nearly
        # meaningless for alerting.
        relative = per_pixel_m / distance_m if distance_m > 0 else 1.0
        confidence = max(0.0, min(1.0, 1.0 - (relative / 0.10)))

        # Beyond the useful range, taper toward zero regardless.
        if distance_m > _MAX_USEFUL_M:
            confidence *= 0.1

        return round(confidence, 3)

    def estimate(
        self,
        det: Detection,
        frame_shape: tuple[int, int],
        gps: GpsFix | None,
    ) -> DistanceEstimate:
        frame_h, frame_w = frame_shape
        _, y_bottom = det.bbox.bottom_center

        distance = self._distance_for_row(y_bottom, frame_h, frame_w)

        if distance is None:
            # Above the horizon. Report it honestly with zero confidence
            # rather than inventing a plausible number — a fabricated distance
            # here would drive a real alert.
            return DistanceEstimate(
                distance_m=float("inf"),
                time_to_pothole_s=None,
                confidence=0.0,
                method=self.method,
                assumptions=[
                    *self._assumptions,
                    "detection is at or above the horizon; no road-plane "
                    "intersection exists",
                ],
            )

        confidence = self._confidence(distance, y_bottom, frame_h, frame_w)

        time_to_pothole = None
        assumptions = list(self._assumptions)
        if gps is not None and gps.speed_kmh is not None and gps.speed_kmh > 0:
            speed_ms = gps.speed_kmh / 3.6
            time_to_pothole = distance / speed_ms
            assumptions.append(f"speed {gps.speed_kmh:.0f} km/h from GPS")

        return DistanceEstimate(
            distance_m=round(distance, 2),
            time_to_pothole_s=round(time_to_pothole, 2) if time_to_pothole else None,
            confidence=confidence,
            method=self.method,
            assumptions=assumptions,
        )
