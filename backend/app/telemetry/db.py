"""Database engine, sessions and migrations.

PostgreSQL when TELEMETRY_DATABASE_URL is set (postgresql+psycopg://...), otherwise a local SQLite file as a
development fallback. The schema is managed by Alembic (app/telemetry/migrations); `upgrade()` brings any database to
the latest revision and runs at startup.

CLI (from backend/):
    python -m app.telemetry.db upgrade      # apply migrations to the configured database
    python -m app.telemetry.db current      # show the database's revision
    python -m app.telemetry.db purge        # delete measurements older than the retention window
"""

import argparse
import logging
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def database_url(settings: Settings) -> tuple[str, str]:
    """(url, kind) where kind is "postgresql" or "sqlite (development fallback)". The URL is never logged."""
    if settings.telemetry_database_url is not None:
        url = settings.telemetry_database_url.get_secret_value()
        return url, url.split(":", 1)[0].split("+", 1)[0]
    path = settings.resolve(settings.telemetry_sqlite_fallback)
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}", "sqlite (development fallback)"


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            if ":memory:" not in url:
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")  # WAL + NORMAL: safe on app crash, much faster commits
            cur.close()

        return engine
    # fail fast when the server is down: without a timeout, startup can hang for minutes (Windows SYN retries)
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5, connect_args={"connect_timeout": 5})


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


def _alembic_config(engine: Engine):
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.attributes["engine"] = engine
    return cfg


def upgrade(engine: Engine, revision: str = "head") -> None:
    from alembic import command

    command.upgrade(_alembic_config(engine), revision)


def current_revision(engine: Engine) -> str | None:
    from alembic.runtime.migration import MigrationContext

    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["upgrade", "current", "purge"])
    a = p.parse_args()
    settings = get_settings()
    url, kind = database_url(settings)
    engine = make_engine(url)
    print(f"Telemetry database: {kind}")
    if a.action == "upgrade":
        upgrade(engine)
        print(f"At revision {current_revision(engine)}")
    elif a.action == "current":
        print(current_revision(engine))
    else:
        from app.telemetry.store import TelemetryStore

        n = TelemetryStore(session_factory(engine)).purge(settings.telemetry_retention_days)
        print(f"Deleted {n} measurements older than {settings.telemetry_retention_days} days")


if __name__ == "__main__":
    main()
