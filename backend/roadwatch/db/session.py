"""Database engine and session management.

Two very different stores live behind this module:

- **PostgreSQL + PostGIS** on the server, for spatial dedup and radius queries.
- **SQLite** on the edge, for the local report store and sync queue, which must
  work with zero network (FR-SYN-02).

The edge store deliberately does not use the PostGIS models above — an edge
device holds a queue, not a geospatial index.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from roadwatch.core.logging import get_logger

log = get_logger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def init_engine(database_url: str, *, echo: bool = False) -> Engine:
    """Create the engine and session factory. Call once at startup."""
    global _engine, _SessionFactory

    connect_args: dict[str, object] = {}
    if database_url.startswith("sqlite"):
        # SQLite defaults to serialising all access to one thread; the edge
        # runtime writes from the pipeline thread and syncs from another.
        connect_args["check_same_thread"] = False

    _engine = create_engine(
        database_url,
        echo=echo,
        pool_pre_ping=True,
        connect_args=connect_args,
    )

    if database_url.startswith("sqlite"):
        _enable_sqlite_durability(_engine)

    log.info("database_engine_initialised", dialect=_engine.dialect.name)
    _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def _enable_sqlite_durability(engine: Engine) -> None:
    """Configure SQLite so a power loss does not cost us reports.

    A vehicle device loses power abruptly and often — the engine stops, the
    cable is pulled. WAL plus ``synchronous=FULL`` is the combination that
    keeps a committed report committed (FR-SYN-01, FR-SYN-02).
    """

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection: object, _record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=FULL")
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("init_engine() has not been called")
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on any exception."""
    if _SessionFactory is None:
        raise RuntimeError("init_engine() has not been called")

    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        # Re-raised with its context intact; callers decide what to do
        # (NFR-OBS-05).
        raise
    finally:
        session.close()


def edge_database_url(storage_root: Path) -> str:
    """SQLite URL for the on-vehicle store."""
    storage_root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(storage_root / 'roadwatch-edge.db').as_posix()}"
