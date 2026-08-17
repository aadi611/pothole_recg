"""Contract invariant tests.

These guard the properties other modules rely on. ``bottom_center`` in
particular is the basis of every distance estimate — if it is wrong, every
alert in the system fires at the wrong moment.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from roadwatch.core.contracts import (
    BBox,
    Detection,
    DistanceEstimate,
    GpsFix,
    HazardClass,
    ReportStatus,
    TrackState,
)


def test_bbox_dimensions() -> None:
    box = BBox(x1=10, y1=20, x2=110, y2=70)
    assert box.width == 100
    assert box.height == 50
    assert box.area_px == 5000


def test_bottom_center_is_the_ground_contact_point() -> None:
    """Horizontal midpoint, bottom edge — where the defect meets the road."""
    box = BBox(x1=100, y1=200, x2=300, y2=400)
    assert box.bottom_center == (200.0, 400.0)


def test_bottom_center_uses_y2_not_y1() -> None:
    """Regression guard: using the top edge would over-estimate distance.

    A box whose top is used instead of its bottom projects to a point further
    up the image, which the planar estimator reads as further away — the
    driver would be warned late, or not at all.
    """
    box = BBox(x1=0, y1=0, x2=100, y2=500)
    _, y = box.bottom_center
    assert y == 500.0


def test_detection_rejects_confidence_above_one() -> None:
    with pytest.raises(ValidationError):
        Detection(
            frame_id=1,
            timestamp=datetime.now(UTC),
            hazard_class=HazardClass.POTHOLE,
            confidence=1.4,
            bbox=BBox(x1=0, y1=0, x2=1, y2=1),
            model_version="test",
        )


def test_detection_is_frozen() -> None:
    """A Detection is a record of what the model said. Nothing may edit it."""
    det = Detection(
        frame_id=1,
        timestamp=datetime.now(UTC),
        hazard_class=HazardClass.POTHOLE,
        confidence=0.9,
        bbox=BBox(x1=0, y1=0, x2=1, y2=1),
        model_version="test",
    )
    with pytest.raises(ValidationError):
        det.confidence = 0.1  # type: ignore[misc]


def test_distance_estimate_carries_method_and_assumptions() -> None:
    """Never present an estimate as a measurement (ethical guardrail 2)."""
    est = DistanceEstimate(
        distance_m=11.3,
        confidence=0.62,
        method="planar",
        assumptions=["flat road", "fixed camera pitch"],
    )
    assert est.method == "planar"
    assert est.assumptions


def test_gps_rejects_out_of_range_coordinates() -> None:
    with pytest.raises(ValidationError):
        GpsFix(latitude=91.0, longitude=0.0, timestamp=datetime.now(UTC), source="test")
    with pytest.raises(ValidationError):
        GpsFix(latitude=0.0, longitude=181.0, timestamp=datetime.now(UTC), source="test")


def test_gps_heading_wraps_below_360() -> None:
    with pytest.raises(ValidationError):
        GpsFix(
            latitude=28.57,
            longitude=77.32,
            heading_deg=360.0,
            timestamp=datetime.now(UTC),
            source="test",
        )


def test_pending_manual_and_sent_to_authority_are_distinct() -> None:
    """The project's most important rule, encoded as a type.

    PENDING_MANUAL means no channel accepted the report. SENT_TO_AUTHORITY
    means one did. Collapsing them would let the system claim to have filed
    civic complaints it never filed (FR-AUTH-05).
    """
    assert ReportStatus.PENDING_MANUAL != ReportStatus.SENT_TO_AUTHORITY
    assert ReportStatus.PENDING_MANUAL.value == "pending_manual"
    assert ReportStatus.SENT_TO_AUTHORITY.value == "sent_to_authority"


def test_track_state_machine_covers_the_specified_path() -> None:
    for state in ("UNKNOWN", "DETECTED", "APPROACHING", "VERY_NEAR", "CAPTURED", "REPORTED", "LOST"):
        assert hasattr(TrackState, state)
