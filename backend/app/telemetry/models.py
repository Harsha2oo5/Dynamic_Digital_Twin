"""Telemetry database schema (SQLAlchemy 2.0). PostgreSQL in deployment; SQLite for tests and the dev fallback.

BTS site -> equipment -> sensors -> measurements. Equipment IDs (e.g. BAT-001) are stable, operator-assigned
identities; they are NOT tracker IDs. Visual tracks can be linked to equipment only explicitly
(equipment_visual_links), never automatically.
"""

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetimes on every backend (SQLite drops the offset, so it is re-attached on read)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime; telemetry timestamps must be timezone-aware")
        value = value.astimezone(UTC)
        return value.replace(tzinfo=None) if dialect.name == "sqlite" else value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


ID = String(64)
BigId = BigInteger().with_variant(Integer, "sqlite")  # SQLite autoincrement needs INTEGER PRIMARY KEY


class Base(DeclarativeBase):
    pass


class Site(Base):
    __tablename__ = "sites"
    site_id: Mapped[str] = mapped_column(ID, primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    location: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Equipment(Base):
    __tablename__ = "equipment"
    equipment_id: Mapped[str] = mapped_column(ID, primary_key=True)
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.site_id"), index=True)
    category: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str] = mapped_column(String(200))
    zone: Mapped[str | None] = mapped_column(String(100))
    data_source: Mapped[str] = mapped_column(String(16), default="physical")  # physical | simulated
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class EquipmentVisualLink(Base):
    """Explicit, operator-made link between a stable equipment item and a visual observation (session track)."""

    __tablename__ = "equipment_visual_links"
    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    equipment_id: Mapped[str] = mapped_column(ForeignKey("equipment.equipment_id", ondelete="CASCADE"), index=True)
    object_id: Mapped[str] = mapped_column(String(128))  # inventory object id (source:track_id); session-scoped
    class_name: Mapped[str | None] = mapped_column(String(64))
    linked_by: Mapped[str | None] = mapped_column(String(100))
    note: Mapped[str | None] = mapped_column(Text)
    linked_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Sensor(Base):
    __tablename__ = "sensors"
    sensor_id: Mapped[str] = mapped_column(ID, primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    sensor_type: Mapped[str] = mapped_column(String(32), index=True)
    protocol: Mapped[str] = mapped_column(String(16))
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.site_id"), index=True)
    equipment_id: Mapped[str | None] = mapped_column(ForeignKey("equipment.equipment_id", ondelete="SET NULL"),
                                                     index=True)
    category: Mapped[str] = mapped_column(String(32), index=True)  # equipment category, or "site"
    zone: Mapped[str | None] = mapped_column(String(100))
    data_source: Mapped[str] = mapped_column(String(16), default="physical")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    metrics: Mapped[list] = mapped_column(JSON, default=list)  # metric names this sensor may report
    sampling_interval_s: Mapped[float] = mapped_column(Float, default=10.0)
    stale_after_s: Mapped[float | None] = mapped_column(Float)  # default 3 x sampling interval
    offline_after_s: Mapped[float | None] = mapped_column(Float)  # default 10 x sampling interval (min 60 s)
    config: Mapped[dict] = mapped_column(JSON, default=dict)  # non-secret settings only
    last_seen_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class Measurement(Base):
    __tablename__ = "measurements"
    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    sensor_id: Mapped[str] = mapped_column(ForeignKey("sensors.sensor_id", ondelete="CASCADE"))
    equipment_id: Mapped[str | None] = mapped_column(ID)  # snapshot at ingest time (re-association keeps history)
    site_id: Mapped[str] = mapped_column(ID)
    metric: Mapped[str] = mapped_column(String(48))
    value: Mapped[float | None] = mapped_column(Float)  # canonical unit; NULL when missing or invalid
    unit: Mapped[str] = mapped_column(String(16))
    ts: Mapped[datetime] = mapped_column(UTCDateTime)  # measurement time (sensor clock or server time)
    received_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    data_source: Mapped[str] = mapped_column(String(16))  # physical | simulated
    protocol: Mapped[str] = mapped_column(String(16))
    quality: Mapped[str] = mapped_column(String(16))  # good | missing | invalid | out_of_order
    quality_note: Mapped[str | None] = mapped_column(String(200))
    message_id: Mapped[str | None] = mapped_column(String(128))
    raw_value: Mapped[str | None] = mapped_column(String(64))
    raw_unit: Mapped[str | None] = mapped_column(String(16))

    __table_args__ = (
        UniqueConstraint("sensor_id", "metric", "ts", name="uq_measurement_sensor_metric_ts"),
        Index("ix_measurement_equipment_metric_ts", "equipment_id", "metric", "ts"),
        Index("ix_measurement_metric_ts", "metric", "ts"),
        Index("ix_measurement_ts_brin", "ts", postgresql_using="brin"),
    )


class SensorStatusEvent(Base):
    """Connection-state transitions (online / stale / offline), for availability history."""

    __tablename__ = "sensor_status_events"
    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    sensor_id: Mapped[str] = mapped_column(ForeignKey("sensors.sensor_id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(16))
    at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    rule_id: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(16))  # warning | critical
    state: Mapped[str] = mapped_column(String(16), index=True)  # active | cleared
    site_id: Mapped[str] = mapped_column(ID)
    sensor_id: Mapped[str | None] = mapped_column(ID, index=True)
    equipment_id: Mapped[str | None] = mapped_column(ID, index=True)
    metric: Mapped[str | None] = mapped_column(String(48))
    value: Mapped[float | None] = mapped_column(Float)
    peak_value: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)
    message: Mapped[str] = mapped_column(Text)
    data_source: Mapped[str] = mapped_column(String(16))
    opened_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    last_triggered_at: Mapped[datetime] = mapped_column(UTCDateTime)
    cleared_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    acknowledged_by: Mapped[str | None] = mapped_column(String(100))
    ack_note: Mapped[str | None] = mapped_column(Text)
