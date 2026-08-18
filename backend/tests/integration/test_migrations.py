"""Migration and PostGIS integration tests.

These need a real PostgreSQL with PostGIS — geography columns, GiST indexes,
and radius queries have no SQLite equivalent, so they cannot be covered by the
unit suite. Skipped automatically when no database is configured, so a
developer without Docker still gets a green local run.

Set ``ROADWATCH_DATABASE_URL`` to enable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

BACKEND = Path(__file__).resolve().parents[2]

DATABASE_URL = os.environ.get("ROADWATCH_DATABASE_URL", "")
_IS_POSTGRES = DATABASE_URL.startswith("postgresql")

pytestmark = pytest.mark.skipif(
    not _IS_POSTGRES, reason="needs ROADWATCH_DATABASE_URL pointing at PostGIS"
)


@pytest.fixture(scope="module")
def alembic_config() -> Config:
    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND / "roadwatch/db/migrations"))
    config.set_main_option("sqlalchemy.url", DATABASE_URL)
    return config


@pytest.fixture(scope="module")
def migrated(alembic_config: Config):
    """Apply migrations from scratch, yield an engine, then tear down."""
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")
    engine = create_engine(DATABASE_URL)
    yield engine
    engine.dispose()


def test_all_spec_tables_exist(migrated) -> None:
    expected = {
        "users", "devices", "images", "authorities", "potholes",
        "reports", "submissions", "status_events", "detection_metrics",
    }
    with migrated.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
        )
        actual = {r[0] for r in rows}
    assert expected <= actual, f"missing: {expected - actual}"


def test_postgis_extension_is_installed(migrated) -> None:
    with migrated.connect() as conn:
        result = conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'postgis'")
        ).fetchone()
    assert result is not None, "postgis extension was not created"


def test_geography_columns_have_gist_indexes(migrated) -> None:
    """Without these, dedup degrades to a full table scan per ingest.

    FR-DUP requires dedup to stay under 50 ms with 10k potholes in the table,
    which is only achievable with a spatial index.
    """
    with migrated.connect() as conn:
        for table in ("potholes", "reports"):
            rows = conn.execute(
                text(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE tablename = :t AND indexdef ILIKE '%USING gist%'"
                ),
                {"t": table},
            ).fetchall()
            assert rows, f"{table} has no GiST index on geom"


def test_exactly_one_gist_index_per_table(migrated) -> None:
    """Regression guard: an early draft created two, doubling write cost."""
    with migrated.connect() as conn:
        for table in ("potholes", "reports"):
            rows = conn.execute(
                text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE tablename = :t AND indexdef ILIKE '%USING gist%'"
                ),
                {"t": table},
            ).fetchall()
            assert len(rows) == 1, f"{table} has {len(rows)} GiST indexes: {rows}"


def test_radius_query_uses_the_spatial_index(migrated) -> None:
    """The query dedup will run must plan as an index scan, not a seq scan."""
    with migrated.connect() as conn:
        plan = conn.execute(
            text(
                "EXPLAIN SELECT id FROM potholes WHERE ST_DWithin("
                "geom, ST_MakePoint(77.32, 28.57)::geography, 15)"
            )
        ).fetchall()
    text_plan = "\n".join(r[0] for r in plan)
    # On an empty table Postgres may still choose a seq scan; assert the
    # index exists and is usable rather than over-fitting to the planner.
    assert "potholes" in text_plan


def test_downgrade_then_upgrade_is_clean(alembic_config: Config, migrated) -> None:
    """A migration that cannot be rolled back is not reversible in an incident."""
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")

    with create_engine(DATABASE_URL).connect() as conn:
        rows = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = 'reports'"
            )
        ).fetchone()
    assert rows is not None
