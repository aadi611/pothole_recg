"""Schema tests.

These assert the structural properties other requirements depend on — the
idempotency key, the audit trail, the nullable columns that mean "not
measured". They run against the metadata, so they need no database.

Executing the migration against real PostGIS is an integration test and is
covered separately once Docker is available; geography columns and GiST
indexes cannot be exercised on SQLite.
"""

from __future__ import annotations

from roadwatch.db.models import (
    Authority,
    Base,
    DetectionMetric,
    Device,
    Image,
    Pothole,
    Report,
    StatusEvent,
    Submission,
    User,
)

SPEC_TABLES = {
    "users",
    "devices",
    "images",
    "authorities",
    "potholes",
    "reports",
    "submissions",
    "status_events",
    "detection_metrics",
}


def test_every_spec_table_exists() -> None:
    """spec/03_DATA_CONTRACTS.md section 5 lists nine server-side tables."""
    assert set(Base.metadata.tables) == SPEC_TABLES


def test_report_id_is_the_primary_key() -> None:
    """Idempotent ingest depends on this (FR-API-02).

    The edge mints report_id before sync. Making it the primary key means a
    resubmitted offline report collides instead of duplicating, so replaying
    a batch three times cannot inflate the row count.
    """
    pk = list(Report.__table__.primary_key.columns)
    assert len(pk) == 1
    assert pk[0].name == "id"


def test_distance_at_capture_is_nullable() -> None:
    """Null means "not measured", never zero (FR-PRX-06).

    Below min_distance_confidence the pipeline records no distance. A
    non-nullable column would force a fabricated 0.0 into a civic report.
    """
    assert Report.__table__.c.distance_at_capture_m.nullable is True


def test_severity_reason_is_required() -> None:
    """Never emit a severity without its reason (spec section 7)."""
    assert Report.__table__.c.severity_reason.nullable is False
    assert Report.__table__.c.severity.nullable is False


def test_authority_match_reason_exists_on_reports() -> None:
    """The resolver stores why it matched, or why it did not (FR-AUTH-02/03)."""
    assert "authority_match_reason" in Report.__table__.c


def test_submission_records_success_and_reference() -> None:
    """The audit trail behind FR-AUTH-05.

    SENT_TO_AUTHORITY is only legitimate when a row here says a channel
    actually accepted the report.
    """
    cols = Submission.__table__.c
    assert cols.success.nullable is False
    assert "external_reference" in cols
    assert "response_json" in cols
    assert "attempt_no" in cols


def test_geography_columns_use_srid_4326() -> None:
    for model in (Pothole, Report):
        geom = model.__table__.c.geom
        assert geom.type.srid == 4326, f"{model.__tablename__}.geom wrong SRID"
        assert geom.nullable is False


def test_exactly_one_gist_index_per_geography_column() -> None:
    """GeoAlchemy2 emits its own GiST index per geography column.

    Regression guard: the first migration draft declared a second one by hand,
    which would have doubled write cost on every ingest for no benefit. There
    must be exactly one, and it must be GiST.
    """
    for model in (Pothole, Report):
        geom_indexes = [
            i for i in model.__table__.indexes if list(i.columns.keys()) == ["geom"]
        ]
        assert len(geom_indexes) == 1, (
            f"{model.__tablename__} has {len(geom_indexes)} geom indexes, expected 1"
        )
        assert geom_indexes[0].dialect_options["postgresql"]["using"] == "gist"


def test_hot_query_columns_are_indexed() -> None:
    """Indexes required by spec/03_DATA_CONTRACTS.md section 5."""
    report_indexed = {
        c.name for c in Report.__table__.c if c.index
    } | {c.name for idx in Report.__table__.indexes for c in idx.columns}
    for col in ("detected_at", "status", "pothole_id"):
        assert col in report_indexed, f"reports.{col} is not indexed"

    submission_indexed = {c.name for c in Submission.__table__.c if c.index}
    assert "report_id" in submission_indexed


def test_reports_cascade_from_device_but_not_from_pothole() -> None:
    """Deleting a device removes its reports (NFR-PRV-06 data deletion).

    Dedup may detach a report from a pothole, which must not delete the
    report — the observation still happened.
    """
    fks = {fk.column.table.name: fk.ondelete for fk in Report.__table__.foreign_keys}
    assert fks["devices"] == "CASCADE"
    assert fks["potholes"] == "SET NULL"


def test_models_are_importable_and_distinct() -> None:
    models = [
        User, Device, Image, Authority, Pothole, Report,
        Submission, StatusEvent, DetectionMetric,
    ]
    assert len({m.__tablename__ for m in models}) == len(models)
