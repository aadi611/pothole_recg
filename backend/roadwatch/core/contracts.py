"""Cross-module payload types.

Every value that crosses a module boundary is defined here (see
``spec/03_DATA_CONTRACTS.md``). No bare dicts between modules — a dict that
travels between two modules is an undocumented interface that drifts.

Change this file *before* changing the code that uses it, and update the spec
in the same commit.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, computed_field


class HazardClass(str, Enum):
    """Road defect taxonomy.

    Only POTHOLE is active in the shipped class map; the rest are defined so
    they can be enabled via config without a code change (FR-DET-03).
    """

    POTHOLE = "pothole"
    CRACK = "crack"
    SURFACE_DAMAGE = "surface_damage"
    MANHOLE = "manhole"
    DEBRIS = "debris"
    UNKNOWN = "unknown"


class BBox(BaseModel):
    """Axis-aligned box in pixel coordinates, origin top-left."""

    model_config = ConfigDict(frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @computed_field  # type: ignore[prop-decorator]
    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def bottom_center(self) -> tuple[float, float]:
        """The ground-contact point.

        This is the load-bearing assumption of planar distance estimation
        (FR-PRX-02): the bottom edge of the box is where the defect meets the
        road surface, so it can be projected onto the ground plane. It is why
        detection must emit boxes and a classifier cannot serve this system.

        Wrong when the defect is occluded at its lower edge, or when the box
        is clipped by the frame boundary. See docs/DISTANCE_LIMITATIONS.md.
        """
        return ((self.x1 + self.x2) / 2, self.y2)

    @property
    def area_px(self) -> float:
        return self.width * self.height


class Detection(BaseModel):
    """One model output on one frame. No tracking, no distance."""

    model_config = ConfigDict(frozen=True)

    frame_id: int
    timestamp: datetime
    hazard_class: HazardClass
    confidence: float = Field(ge=0, le=1)
    bbox: BBox
    mask_rle: str | None = None
    model_version: str


class DistanceEstimate(BaseModel):
    """An *estimate*, never a measurement.

    ``confidence`` and ``assumptions`` are not decoration — the UI is required
    to surface them (FR-PRX-06, ethical guardrail 2), and below
    ``min_distance_confidence`` the pipeline warns but does not auto-capture.
    """

    model_config = ConfigDict(frozen=True)

    distance_m: float
    time_to_pothole_s: float | None = None
    confidence: float = Field(ge=0, le=1)
    method: str
    assumptions: list[str] = Field(default_factory=list)


class TrackState(str, Enum):
    """Lifecycle of one tracked defect (FR-TRK-03).

    ``UNKNOWN -> DETECTED -> APPROACHING -> VERY_NEAR -> CAPTURED -> REPORTED``
    with ``LOST`` reachable from any state. A LOST track that already reached
    CAPTURED still completes its report (FR-TRK-04).
    """

    UNKNOWN = "UNKNOWN"
    DETECTED = "DETECTED"
    APPROACHING = "APPROACHING"
    VERY_NEAR = "VERY_NEAR"
    CAPTURED = "CAPTURED"
    REPORTED = "REPORTED"
    LOST = "LOST"


class Track(BaseModel):
    """A physical defect followed across frames."""

    track_id: str
    hazard_class: HazardClass
    state: TrackState
    first_seen: datetime
    last_seen: datetime
    frames_seen: int
    peak_confidence: float = Field(ge=0, le=1)
    latest: Detection
    distance: DistanceEstimate | None = None
    estimated_size_m2: float | None = None


class GpsFix(BaseModel):
    """A position sample.

    ``is_stale`` means the fix is older than the configured max age — the
    report is still created (a pothole seen in a tunnel is still a pothole)
    but it is flagged for review rather than silently trusted (FR-GEO-04).
    """

    model_config = ConfigDict(frozen=True)

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    accuracy_m: float | None = None
    speed_kmh: float | None = None
    heading_deg: float | None = Field(default=None, ge=0, lt=360)
    altitude_m: float | None = None
    timestamp: datetime
    source: str
    is_stale: bool = False


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReportStatus(str, Enum):
    """Report lifecycle.

    Note the deliberate split between PENDING_MANUAL and SENT_TO_AUTHORITY.
    A report reaches SENT_TO_AUTHORITY only when a channel actually accepted
    it and returned a reference (FR-AUTH-05). Everything else is
    PENDING_MANUAL. This is the project's most important rule: a system that
    pretends to have filed civic complaints is worse than one that admits it
    cannot.
    """

    DRAFT = "draft"
    QUEUED = "queued"
    SUBMITTED_PENDING = "submitted_pending"
    PENDING_MANUAL = "pending_manual"
    SENT_TO_AUTHORITY = "sent_to_authority"
    ACKNOWLEDGED = "acknowledged"
    UNDER_REPAIR = "under_repair"
    RESOLVED = "resolved"
    REOPENED = "reopened"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


class LocationContext(BaseModel):
    """Reverse-geocoded address. Resolved server-side, may be empty."""

    city: str | None = None
    district: str | None = None
    state: str | None = None
    country: str | None = None
    road_name: str | None = None
    postcode: str | None = None
    osm_way_id: str | None = None
    road_class: str | None = None
    geocoder: str | None = None
    geocoded_at: datetime | None = None


class PotholeReport(BaseModel):
    """One confirmed defect observed by one device on one pass."""

    report_id: str
    pothole_id: str | None = None
    device_id: str
    camera_id: str
    detected_at: datetime
    gps: GpsFix
    location: LocationContext = Field(default_factory=LocationContext)
    hazard_class: HazardClass
    detection_confidence: float = Field(ge=0, le=1)
    distance_at_capture_m: float | None = None
    estimated_size_m2: float | None = None
    severity: Severity
    severity_reason: str = Field(min_length=1)
    image_path: str
    crop_path: str | None = None
    image_sha256: str
    redaction_applied: list[str] = Field(default_factory=list)
    model_version: str
    app_version: str
    status: ReportStatus
    authority_id: str | None = None
    authority_match_reason: str | None = None
    submission_attempts: int = 0
    created_at: datetime
    updated_at: datetime


class Pothole(BaseModel):
    """Server-side aggregate: one physical defect, many reports.

    ``confidence`` rises with ``distinct_device_count``, not ``report_count``
    (FR-DUP-04) — one car passing daily is weaker evidence than ten cars
    passing once.
    """

    pothole_id: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    location: LocationContext = Field(default_factory=LocationContext)
    hazard_class: HazardClass
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    report_count: int = 0
    distinct_device_count: int = 0
    first_reported_at: datetime
    last_seen_at: datetime
    status: ReportStatus
    authority_id: str | None = None
    latest_image_path: str | None = None


class SubmissionResult(BaseModel):
    """Outcome of one attempt to file a report with an authority.

    ``success=True`` requires a real accepted delivery. There is no code path
    that may construct this with ``success=True`` and no ``external_reference``
    for channels that issue one — asserted by a dedicated test (NFR-TST-06).
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    status: ReportStatus
    external_reference: str | None = None
    message: str
    retryable: bool
    attempted_at: datetime
