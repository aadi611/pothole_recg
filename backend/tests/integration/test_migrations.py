
"""Migration and PostGIS integration tests.

These tests require a real PostgreSQL instance with PostGIS enabled.
SQLite cannot validate geography columns, GiST indexes, or spatial queries.

Set ROADWATCH_DATABASE_URL to enable these tests.

WARNING:
The migration fixture downgrades the configured database to base before
reapplying migrations. Use a dedicated, disposable test database.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND = Path(__file__).resolve().parents[2]

DATABASE_URL = os.getenv("ROADWATCH_DATABASE_URL", "")
IS_POSTGRES = DATABASE_URL.startswith(("postgresql://", "postgresql+"))

EXPECTED_TABLES = {
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

SPATIAL_TABLES = ("potholes", "reports")

POSTGIS_EXTENSION = "postgis"


pytestmark = pytest.mark.skipif(
    not IS_POSTGRES,
    reason="Requires ROADWATCH_DATABASE_URL pointing to a PostGIS database",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def alembic_config() -> Config:
    """Configure Alembic to use the integration-test database."""
    config = Config(str(BACKEND / "alembic.ini"))

    config.set_main_option(
        "script_location",
        str(BACKEND / "roadwatch/db/migrations"),
    )

    # Escape percent signs because ConfigParser uses interpolation.
    config.set_main_option(
        "sqlalchemy.url",
        DATABASE_URL.replace("%", "%%"),
    )

    return config


@pytest.fixture(scope="module")
def migrated(alembic_config: Config) -> Generator[Engine, None, None]:
    """Run migrations from scratch and provide a database engine.

    The configured database is reset to the Alembic base revision before
    migrations are applied.
    """
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    try:
        yield engine
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def fetch_all(engine: Engine, query: str, params: dict | None = None):
    """Execute a query and return all rows."""
    with engine.connect() as connection:
        return connection.execute(
            text(query),
            params or {},
        ).fetchall()


def fetch_one(engine: Engine, query: str, params: dict | None = None):
    """Execute a query and return the first row."""
    with engine.connect() as connection:
        return connection.execute(
            text(query),
            params or {},
        ).fetchone()


def get_gist_indexes(engine: Engine, table: str):
    """Return GiST index definitions for a given table."""
    return fetch_all(
        engine,
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename = :table
          AND indexdef ILIKE '%USING gist%'
        """,
        {"table": table},
    )


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------

def test_all_spec_tables_exist(migrated: Engine) -> None:
    """Verify that all specification tables are created."""
    rows = fetch_all(
        migrated,
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        """,
    )

    actual_tables = {row[0] for row in rows}
    missing_tables = EXPECTED_TABLES - actual_tables

    assert not missing_tables, (
        f"Missing specification tables: {sorted(missing_tables)}"
    )


def test_postgis_extension_is_installed(migrated: Engine) -> None:
    """Verify that the PostGIS extension is installed."""
    result = fetch_one(
        migrated,
        """
        SELECT extname
        FROM pg_extension
        WHERE extname = :extension
        """,
        {"extension": POSTGIS_EXTENSION},
    )

    assert result is not None, "PostGIS extension was not installed"


# ---------------------------------------------------------------------------
# Spatial index tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("table", SPATIAL_TABLES)
def test_geography_columns_have_gist_indexes(
    migrated: Engine,
    table: str,
) -> None:
    """Verify that each spatial table has a GiST index on geom."""
    indexes = get_gist_indexes(migrated, table)

    geom_indexes = [
        index
        for index in indexes
        if "geom" in index[1].lower()
    ]

    assert geom_indexes, (
        f"{table} does not have a GiST index covering geom"
    )


@pytest.mark.parametrize("table", SPATIAL_TABLES)
def test_exactly_one_gist_index_per_table(
    migrated: Engine,
    table: str,
) -> None:
    """Prevent duplicate GiST indexes from increasing write overhead."""
    indexes = get_gist_indexes(migrated, table)

    assert len(indexes) == 1, (
        f"{table} has {len(indexes)} GiST indexes; "
        f"expected exactly one. Found: {indexes}"
    )

    assert "geom" in indexes[0][1].lower(), (
        f"The GiST index on {table} does not cover geom: {indexes[0][1]}"
    )


# ---------------------------------------------------------------------------
# Spatial query performance tests
# ---------------------------------------------------------------------------

def test_radius_query_uses_the_spatial_index(migrated: Engine) -> None:
    """Verify that PostgreSQL can use the spatial index for radius queries.

    Disabling sequential scans for this EXPLAIN test makes the planner
    consider the available GiST index even when the table is empty.

    This validates index usability, not actual production query latency.
    """
    with migrated.begin() as connection:
        connection.execute(text("SET LOCAL enable_seqscan = off"))

        plan = connection.execute(
            text(
                """
                EXPLAIN
                SELECT id
                FROM potholes
                WHERE ST_DWithin(
                    geom,
                    ST_SetSRID(ST_MakePoint(77.32, 28.57), 4326)::geography,
                    15
                )
                """
            )
        ).fetchall()

    execution_plan = "\n".join(row[0] for row in plan)

    assert (
        "Index Scan" in execution_plan
        or "Bitmap Index Scan" in execution_plan
    ), (
        "Radius query did not use a spatial index.\n"
        f"Execution plan:\n{execution_plan}"
    )


# ---------------------------------------------------------------------------
# Migration rollback tests
# ---------------------------------------------------------------------------

def test_downgrade_then_upgrade_is_clean(
    alembic_config: Config,
    migrated: Engine,
) -> None:
    """Verify that migrations can be rolled back and reapplied."""
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")

    result = fetch_one(
        migrated,
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = 'reports'
        """,
    )

    assert result is not None, (
        "The reports table was not recreated after migration rollback"
    )
