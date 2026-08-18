"""SQLAlchemy ORM models (spec/03_DATA_CONTRACTS.md section 5).

**This package is the only place SQLAlchemy may be imported** (architecture
rule 4), enforced by ``scripts/check_boundaries.py``. Everything else works
with the Pydantic contracts in ``core/contracts.py``; conversion happens at
this boundary.

Coordinates are PostGIS ``GEOGRAPHY(POINT, 4326)`` with GiST indexes, because
every proximity query in this system is a radius search, and a brute-force
distance loop over a table of potholes would not survive a real city
(FR-GEO-08, FR-DUP-01).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from geoalchemy2 import Geography
from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base. ``create_all()`` is for tests only (NFR-DEV-03)."""


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    # argon2 (NFR-SEC-08). Never log this column.
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    devices: Mapped[list[Device]] = relationship(back_populates="owner")


class Device(Base):
    __tablename__ = "devices"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(120))
    vehicle_type: Mapped[str | None] = mapped_column(String(40))
    camera_id: Mapped[str | None] = mapped_column(String(64))
    app_version: Mapped[str | None] = mapped_column(String(32))
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    owner_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    owner: Mapped[User | None] = relationship(back_populates="devices")


class Image(Base):
    __tablename__ = "images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # Opaque key into the storage backend, never a client-supplied path
    # (NFR-SEC-06).
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    # What was blurred, e.g. ["faces", "plates"] (NFR-PRV-01).
    redaction_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Authority(Base):
    __tablename__ = "authorities"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    level: Mapped[str] = mapped_column(String(16), nullable=False)
    country: Mapped[str] = mapped_column(String(2), nullable=False)
    state: Mapped[str | None] = mapped_column(String(120))
    # The whole YAML definition, so adding a city needs no schema change
    # (FR-AUTH-01).
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    sla_days: Mapped[int | None] = mapped_column(Integer)
    # Government contact points go stale; the admin UI warns past 12 months
    # (FR-AUTH-10).
    verified_on: Mapped[date | None] = mapped_column(Date)


class Pothole(Base):
    """One physical defect. Many reports may attach to it (FR-DUP-03)."""

    __tablename__ = "potholes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    geom: Mapped[Any] = mapped_column(
        Geography(geometry_type="POINT", srid=4326), nullable=False
    )
    hazard_class: Mapped[str] = mapped_column(String(24), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    # Rises with distinct devices, not raw report count (FR-DUP-04).
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    report_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    distinct_device_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    authority_id: Mapped[str | None] = mapped_column(
        ForeignKey("authorities.id", ondelete="SET NULL")
    )
    location_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    first_reported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    latest_image_id: Mapped[str | None] = mapped_column(
        ForeignKey("images.id", ondelete="SET NULL")
    )

    reports: Mapped[list[Report]] = relationship(back_populates="pothole")

    # GeoAlchemy2 creates the GiST index on `geom` automatically, which is
    # what dedup radius queries use (FR-DUP-01). Declaring a second one here
    # would double write cost on ingest for no benefit.
    __table_args__ = (Index("ix_potholes_last_seen_at", "last_seen_at"),)


class Report(Base):
    """One observation by one device on one pass."""

    __tablename__ = "reports"

    # Edge-minted PTH-YYYYMMDD-NNNNNN, stable before sync (FR-RPT-01). Being
    # the primary key is what makes ingest idempotent (FR-API-02).
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    pothole_id: Mapped[str | None] = mapped_column(
        ForeignKey("potholes.id", ondelete="SET NULL"), index=True
    )
    device_id: Mapped[str] = mapped_column(
        ForeignKey("devices.id", ondelete="CASCADE"), nullable=False, index=True
    )
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    geom: Mapped[Any] = mapped_column(
        Geography(geometry_type="POINT", srid=4326), nullable=False
    )
    speed_kmh: Mapped[float | None] = mapped_column(Float)
    heading_deg: Mapped[float | None] = mapped_column(Float)
    gps_accuracy_m: Mapped[float | None] = mapped_column(Float)
    gps_is_stale: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    hazard_class: Mapped[str] = mapped_column(String(24), nullable=False)
    detection_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    # Null when the distance estimate was below min_distance_confidence
    # (FR-PRX-06). Null means not measured, never zero.
    distance_at_capture_m: Mapped[float | None] = mapped_column(Float)
    estimated_size_m2: Mapped[float | None] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    # Never emit a severity without its reason (spec section 7).
    severity_reason: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    app_version: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    image_id: Mapped[str | None] = mapped_column(ForeignKey("images.id", ondelete="SET NULL"))
    crop_image_id: Mapped[str | None] = mapped_column(ForeignKey("images.id", ondelete="SET NULL"))
    image_sha256: Mapped[str | None] = mapped_column(String(64))
    redaction_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    authority_id: Mapped[str | None] = mapped_column(
        ForeignKey("authorities.id", ondelete="SET NULL")
    )
    # Why this authority matched, or why none did. Stored and displayed
    # (FR-AUTH-02, FR-AUTH-03).
    authority_match_reason: Mapped[str | None] = mapped_column(Text)
    submission_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    pothole: Mapped[Pothole | None] = relationship(back_populates="reports")



class Submission(Base):
    """Every attempt to file a report, including failures (FR-AUTH-06)."""

    __tablename__ = "submissions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    report_id: Mapped[str | None] = mapped_column(
        ForeignKey("reports.id", ondelete="CASCADE"), index=True
    )
    pothole_id: Mapped[str | None] = mapped_column(
        ForeignKey("potholes.id", ondelete="CASCADE")
    )
    authority_id: Mapped[str | None] = mapped_column(
        ForeignKey("authorities.id", ondelete="SET NULL")
    )
    channel: Mapped[str] = mapped_column(String(16), nullable=False)
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # The audit trail behind FR-AUTH-05: a report may only reach
    # SENT_TO_AUTHORITY when a row here records success with a real reference.
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    external_reference: Mapped[str | None] = mapped_column(String(255))
    message: Mapped[str | None] = mapped_column(Text)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    response_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class StatusEvent(Base):
    """Status transition audit (FR-RPT-04, FR-RPT-05)."""

    __tablename__ = "status_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    pothole_id: Mapped[str | None] = mapped_column(
        ForeignKey("potholes.id", ondelete="CASCADE"), index=True
    )
    report_id: Mapped[str | None] = mapped_column(
        ForeignKey("reports.id", ondelete="CASCADE"), index=True
    )
    from_status: Mapped[str | None] = mapped_column(String(24))
    to_status: Mapped[str] = mapped_column(String(24), nullable=False)
    actor: Mapped[str] = mapped_column(String(120), nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DetectionMetric(Base):
    """Per-session pipeline metrics (FR-CAM-06)."""

    __tablename__ = "detection_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    device_id: Mapped[str] = mapped_column(
        ForeignKey("devices.id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    detections: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_fps: Mapped[float | None] = mapped_column(Float)
    avg_inference_ms: Mapped[float | None] = mapped_column(Float)
    # An aggregate total, not a stored breadcrumb trail of where the driver
    # went (NFR-PRV-05).
    distance_km: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint("device_id", "session_id", "window_start", name="uq_metric_window"),
    )
