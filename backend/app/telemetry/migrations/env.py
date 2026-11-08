"""Alembic environment: the engine is passed in by app.telemetry.db (no alembic.ini, no URL in files)."""

from alembic import context

from app.telemetry.models import Base
import app.analytics.models  # noqa: F401  (registers the analytics tables)

target_metadata = Base.metadata


def run_migrations_online() -> None:
    engine = context.config.attributes["engine"]
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata,
                          render_as_batch=connection.dialect.name == "sqlite", compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
        connection.commit()


if context.is_offline_mode():
    raise RuntimeError("Offline migrations are not supported; run app.telemetry.db upgrade")
run_migrations_online()
