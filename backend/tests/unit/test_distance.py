"""Distance estimation tests (FR-PRX-02, FR-PRX-03, FR-PRX-07).

Phase 2 requires planar projection maths to be checked against hand-computed
values (spec/05_BUILD_PHASES.md). The values below were derived independently
from the pinhole equations, not by recording what the code returned.

Reference setup: h=1.25 m, pitch=-8 deg, hfov=78 deg, 1280x720.
  focal_px = 640 / tan(39 deg) = 790.33
  vfov     = 2*atan(360/790.33) = 48.98 deg
  horizon  = 360 - 790.33*tan(8 deg) = 248.9
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from roadwatch.core.config import CameraGeometry
from roadwatch.core.contracts import (
    BBox,
    Detection,
    DistanceEstimate,
    GpsFix,
    HazardClass,
)
from roadwatch.core.errors import CalibrationError
from roadwatch.distance.base import DistanceEstimator
from roadwatch.distance.bbox_scale import BBoxScaleDistanceEstimator
from roadwatch.distance.planar import (
    PlanarDistanceEstimator,
    horizon_row,
    vertical_fov_deg,
)
from roadwatch.distance.smoothing import DistanceSmoother

FRAME = (720, 1280)  # (height, width)


def _camera(**overrides: object) -> CameraGeometry:
    base = {
        "id": "test-cam",
        "mount_height_m": 1.25,
        "pitch_deg": -8.0,
        "horizontal_fov_deg": 78.0,
        "resolution": (1280, 720),
        "calibration_file": None,
    }
    base.update(overrides)
    return CameraGeometry(**base)  # type: ignore[arg-type]


def _det(y_bottom: float, *, x1: float = 600, width: float = 60) -> Detection:
    return Detection(
        frame_id=1,
        timestamp=datetime.now(UTC),
        hazard_class=HazardClass.POTHOLE,
        confidence=0.9,
        bbox=BBox(x1=x1, y1=y_bottom - 40, x2=x1 + width, y2=y_bottom),
        model_version="test",
    )


# --------------------------------------------------------------------------
# Optics helpers
# --------------------------------------------------------------------------


def test_vertical_fov_from_horizontal() -> None:
    """Hand-computed: 2*atan(360 / (640/tan(39deg))) = 48.98 deg."""
    assert vertical_fov_deg(78.0, 1280, 720) == pytest.approx(48.98, abs=0.05)


def test_vertical_fov_uses_pinhole_not_naive_scaling() -> None:
    """Naive aspect scaling (78 * 720/1280 = 43.9) is several degrees wrong."""
    naive = 78.0 * 720 / 1280
    assert abs(vertical_fov_deg(78.0, 1280, 720) - naive) > 4.0


def test_vertical_fov_rejects_bad_input() -> None:
    with pytest.raises(CalibrationError):
        vertical_fov_deg(78.0, 0, 720)
    with pytest.raises(CalibrationError):
        vertical_fov_deg(200.0, 1280, 720)


def test_horizon_row_hand_computed() -> None:
    """360 - 790.33*tan(8 deg) = 248.9."""
    v_fov = vertical_fov_deg(78.0, 1280, 720)
    assert horizon_row(720, 8.0, v_fov) == pytest.approx(248.9, abs=1.0)


# --------------------------------------------------------------------------
# Planar estimator — FR-PRX-02
# --------------------------------------------------------------------------


def test_satisfies_the_protocol() -> None:
    assert isinstance(PlanarDistanceEstimator(_camera()), DistanceEstimator)
    assert isinstance(BBoxScaleDistanceEstimator(_camera()), DistanceEstimator)


@pytest.mark.parametrize(
    ("y_bottom", "expected_m"),
    [
        (360, 8.89),   # image centre: d = h/tan(8 deg)
        (400, 6.49),
        (450, 4.83),
        (500, 3.84),
        (600, 2.69),
        (719, 1.97),   # bottom edge, closest visible ground
    ],
)
def test_distance_matches_hand_computed_geometry(y_bottom: float, expected_m: float) -> None:
    est = PlanarDistanceEstimator(_camera())
    result = est.estimate(_det(y_bottom), FRAME, None)
    assert result.distance_m == pytest.approx(expected_m, abs=0.05)


def test_image_centre_equals_the_closed_form() -> None:
    """At the optical axis the offset is zero, so d = h / tan(pitch)."""
    est = PlanarDistanceEstimator(_camera())
    result = est.estimate(_det(360), FRAME, None)
    assert result.distance_m == pytest.approx(1.25 / math.tan(math.radians(8.0)), abs=0.05)


def test_lower_in_frame_is_closer() -> None:
    """Monotonicity: as a pothole approaches it moves down the frame."""
    est = PlanarDistanceEstimator(_camera())
    distances = [est.estimate(_det(y), FRAME, None).distance_m for y in (300, 400, 500, 600, 700)]
    assert distances == sorted(distances, reverse=True)


def test_above_horizon_returns_infinity_not_a_plausible_number() -> None:
    """Fabricating a distance here would drive a real alert.

    A detection above the horizon is not on the road ahead — it is a wall, a
    vehicle, or a false positive. Reporting a made-up number with real
    confidence would be worse than reporting nothing.
    """
    est = PlanarDistanceEstimator(_camera())
    result = est.estimate(_det(200), FRAME, None)  # horizon is row ~249
    assert result.distance_m == float("inf")
    assert result.confidence == 0.0
    assert any("horizon" in a for a in result.assumptions)


def test_taller_mount_sees_further_at_the_same_pixel_row() -> None:
    low = PlanarDistanceEstimator(_camera(mount_height_m=1.0))
    high = PlanarDistanceEstimator(_camera(mount_height_m=2.0))
    d_low = low.estimate(_det(500), FRAME, None).distance_m
    d_high = high.estimate(_det(500), FRAME, None).distance_m
    assert d_high == pytest.approx(2 * d_low, rel=0.01)


def test_steeper_pitch_sees_less_far() -> None:
    shallow = PlanarDistanceEstimator(_camera(pitch_deg=-5.0))
    steep = PlanarDistanceEstimator(_camera(pitch_deg=-20.0))
    assert (
        steep.estimate(_det(400), FRAME, None).distance_m
        < shallow.estimate(_det(400), FRAME, None).distance_m
    )


def test_upward_pitch_is_rejected_at_construction() -> None:
    """A camera not pointed at the road cannot use this method."""
    with pytest.raises(CalibrationError, match="angled down"):
        PlanarDistanceEstimator(_camera(pitch_deg=5.0))


def test_confidence_falls_with_distance() -> None:
    """Near the horizon one pixel row spans metres of ground.

    Confidence encodes that measured sensitivity, not a vague feeling.
    """
    est = PlanarDistanceEstimator(_camera())
    near = est.estimate(_det(650), FRAME, None)
    far = est.estimate(_det(300), FRAME, None)
    assert near.confidence > far.confidence


def test_every_estimate_carries_assumptions() -> None:
    """Never present an estimate as a measurement (ethical guardrail 2)."""
    est = PlanarDistanceEstimator(_camera())
    result = est.estimate(_det(450), FRAME, None)
    assert result.method == "planar"
    assert any("flat road" in a for a in result.assumptions)
    assert any("height" in a for a in result.assumptions)


def test_frame_shape_is_respected_not_config_resolution() -> None:
    """Distance depends on where the box sits relative to the horizon.

    Using the configured resolution instead of the actual frame shape would
    silently skew every estimate on a camera that returned a different size.
    """
    est = PlanarDistanceEstimator(_camera())
    d_720 = est.estimate(_det(360), (720, 1280), None).distance_m
    d_1080 = est.estimate(_det(540), (1080, 1920), None).distance_m
    # Both are the vertical centre, so both must give the same ground distance.
    assert d_720 == pytest.approx(d_1080, rel=0.02)


# --------------------------------------------------------------------------
# Time to pothole — FR-PRX-05
# --------------------------------------------------------------------------


def _gps(speed_kmh: float | None) -> GpsFix:
    return GpsFix(
        latitude=28.57,
        longitude=77.32,
        speed_kmh=speed_kmh,
        timestamp=datetime.now(UTC),
        source="test",
    )


def test_time_to_pothole_from_gps_speed() -> None:
    est = PlanarDistanceEstimator(_camera())
    result = est.estimate(_det(360), FRAME, _gps(36.0))  # 10 m/s
    assert result.distance_m == pytest.approx(8.89, abs=0.05)
    assert result.time_to_pothole_s == pytest.approx(0.889, abs=0.01)


def test_no_gps_still_yields_a_distance() -> None:
    """GPS must never block the detection pipeline (NFR-REL-01)."""
    est = PlanarDistanceEstimator(_camera())
    result = est.estimate(_det(450), FRAME, None)
    assert result.distance_m > 0
    assert result.time_to_pothole_s is None


def test_stationary_vehicle_has_no_time_to_impact() -> None:
    est = PlanarDistanceEstimator(_camera())
    assert est.estimate(_det(450), FRAME, _gps(0.0)).time_to_pothole_s is None


def test_high_speed_shortens_time_to_impact() -> None:
    """At 80 km/h, 10 m is under half a second (FR-PRX-05)."""
    est = PlanarDistanceEstimator(_camera())
    slow = est.estimate(_det(360), FRAME, _gps(20.0))
    fast = est.estimate(_det(360), FRAME, _gps(80.0))
    assert fast.time_to_pothole_s is not None and slow.time_to_pothole_s is not None
    assert fast.time_to_pothole_s < slow.time_to_pothole_s
    assert fast.time_to_pothole_s < 0.5


# --------------------------------------------------------------------------
# bbox_scale fallback — FR-PRX-03
# --------------------------------------------------------------------------


def test_bbox_scale_confidence_never_permits_capture() -> None:
    """Must stay below the default min_distance_confidence of 0.4.

    Structural, not incidental: a report built on an assumed-width guess would
    put a fabricated measurement into a civic complaint (FR-PRX-06).
    """
    est = BBoxScaleDistanceEstimator(_camera())
    for width in (10, 40, 80, 200, 600):
        result = est.estimate(_det(500, width=width), FRAME, None)
        assert result.confidence < 0.4, f"width={width} gave {result.confidence}"


def test_bbox_scale_wider_box_means_closer() -> None:
    est = BBoxScaleDistanceEstimator(_camera())
    narrow = est.estimate(_det(500, width=20), FRAME, None)
    wide = est.estimate(_det(500, width=200), FRAME, None)
    assert wide.distance_m < narrow.distance_m


def test_bbox_scale_degenerate_box_is_infinite() -> None:
    est = BBoxScaleDistanceEstimator(_camera())
    result = est.estimate(_det(500, width=1), FRAME, None)
    assert result.distance_m == float("inf")
    assert result.confidence == 0.0


def test_bbox_scale_states_its_weak_assumption() -> None:
    est = BBoxScaleDistanceEstimator(_camera())
    result = est.estimate(_det(500), FRAME, None)
    assert result.method == "bbox_scale"
    assert any("assumed pothole width" in a for a in result.assumptions)


# --------------------------------------------------------------------------
# Smoothing — FR-PRX-07
# --------------------------------------------------------------------------


def _estimate(d: float, *, conf: float = 0.8) -> DistanceEstimate:
    return DistanceEstimate(
        distance_m=d, confidence=conf, method="planar", assumptions=["flat road"]
    )


def test_smoothing_averages_over_the_window() -> None:
    s = DistanceSmoother(window=3)
    s.update("t1", _estimate(10.0))
    s.update("t1", _estimate(12.0))
    result = s.update("t1", _estimate(14.0))
    assert result.distance_m == pytest.approx(12.0)


def test_smoothing_suppresses_threshold_flapping() -> None:
    """The reason this exists.

    A raw estimate oscillating around the 10 m warn threshold would cross it
    repeatedly, thrashing the state machine between APPROACHING and DETECTED.
    """
    s = DistanceSmoother(window=5)
    raw = [9.5, 10.5, 9.4, 10.6, 9.6, 10.4]
    smoothed = [s.update("t1", _estimate(d)).distance_m for d in raw]
    # The last few, once the window is full, must sit tightly around 10.
    assert all(abs(d - 10.0) < 0.5 for d in smoothed[3:])


def test_smoothing_is_per_track() -> None:
    """Mixing two potholes would be worse than no smoothing at all."""
    s = DistanceSmoother(window=3)
    s.update("near", _estimate(5.0))
    s.update("far", _estimate(50.0))
    assert s.update("near", _estimate(5.0)).distance_m == pytest.approx(5.0)


def test_smoothing_averages_confidence_too() -> None:
    """One good frame among poor ones must not read as a confident estimate."""
    s = DistanceSmoother(window=4)
    s.update("t1", _estimate(10, conf=0.1))
    s.update("t1", _estimate(10, conf=0.1))
    result = s.update("t1", _estimate(10, conf=0.9))
    assert result.confidence < 0.5


def test_smoothing_passes_infinite_estimates_through_unpoisoned() -> None:
    """An above-horizon reading must not corrupt the running mean."""
    s = DistanceSmoother(window=3)
    s.update("t1", _estimate(10.0))
    passthrough = s.update("t1", _estimate(float("inf")))
    assert passthrough.distance_m == float("inf")
    assert s.update("t1", _estimate(10.0)).distance_m == pytest.approx(10.0)


def test_smoothing_recomputes_time_to_impact_consistently() -> None:
    """The two numbers shown to the driver must agree with each other."""
    s = DistanceSmoother(window=2)
    s.update("t1", DistanceEstimate(distance_m=20.0, time_to_pothole_s=2.0, confidence=0.8, method="planar"))
    result = s.update(
        "t1",
        DistanceEstimate(distance_m=10.0, time_to_pothole_s=1.0, confidence=0.8, method="planar"),
    )
    # Smoothed distance 15 m at the same 10 m/s implies 1.5 s.
    assert result.distance_m == pytest.approx(15.0)
    assert result.time_to_pothole_s == pytest.approx(1.5, abs=0.05)


def test_forget_bounds_memory_over_a_long_drive() -> None:
    """Without this the history dict grows for the whole session (NFR-REL-07)."""
    s = DistanceSmoother()
    for i in range(500):
        s.update(f"track-{i}", _estimate(10.0))
    assert len(s) == 500
    s.retain_only({"track-1", "track-2"})
    assert len(s) == 2


def test_invalid_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="window"):
        DistanceSmoother(window=0)
