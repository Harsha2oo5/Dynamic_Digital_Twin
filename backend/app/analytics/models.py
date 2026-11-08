"""Analytics tables (same database and migration chain as telemetry)."""

from datetime import datetime

from sqlalchemy import JSON, Boolean, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.telemetry.models import ID, Base, BigId, UTCDateTime, utcnow


class Anomaly(Base):
    """One anomaly EPISODE: a continuous (or quickly recurring) abnormal condition of one equipment metric."""

    __tablename__ = "anomalies"
    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    site_id: Mapped[str] = mapped_column(ID)
    equipment_id: Mapped[str | None] = mapped_column(ID)
    sensor_id: Mapped[str | None] = mapped_column(ID)
    category: Mapped[str] = mapped_column(String(32))
    metric: Mapped[str | None] = mapped_column(String(48))  # NULL for multivariate anomalies
    kind: Mapped[str] = mapped_column(String(16))  # univariate | multivariate
    detector: Mapped[str] = mapped_column(String(32))  # detector with the highest score
    detectors: Mapped[list] = mapped_column(JSON, default=list)  # every detector that fired in this episode
    model_id: Mapped[int | None] = mapped_column(Integer)  # anomaly_models.id for learned detectors
    observed: Mapped[float | None] = mapped_column(Float)
    expected: Mapped[float | None] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(16))
    score: Mapped[float] = mapped_column(Float)  # latest score, threshold units (not a probability)
    peak_score: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(16))  # warning | critical (highest reached)
    status: Mapped[str] = mapped_column(String(16))  # open | acknowledged | resolved
    condition_active: Mapped[bool] = mapped_column(Boolean, default=True)
    first_detected_at: Mapped[datetime] = mapped_column(UTCDateTime)
    last_detected_at: Mapped[datetime] = mapped_column(UTCDateTime)
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    occurrences: Mapped[int] = mapped_column(Integer, default=1)
    explanation: Mapped[str] = mapped_column(Text)
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
    data_source: Mapped[str] = mapped_column(String(16))
    acknowledged_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    acknowledged_by: Mapped[str | None] = mapped_column(String(100))
    resolved_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    resolved_by: Mapped[str | None] = mapped_column(String(100))
    resolution_note: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("ix_anomaly_equipment_time", "equipment_id", "first_detected_at"),
        Index("ix_anomaly_status_severity", "status", "severity"),
        Index("ix_anomaly_time", "first_detected_at"),
        Index("ix_anomaly_key", "equipment_id", "sensor_id", "metric", "kind"),
    )


class EquipmentHealth(Base):
    __tablename__ = "equipment_health_history"
    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    equipment_id: Mapped[str] = mapped_column(ID)
    at: Mapped[datetime] = mapped_column(UTCDateTime)
    score: Mapped[float | None] = mapped_column(Float)  # NULL = unknown
    status: Mapped[str] = mapped_column(String(16))  # healthy | warning | degraded | critical | unknown
    coverage: Mapped[float] = mapped_column(Float)
    data_quality: Mapped[float | None] = mapped_column(Float)
    components: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (Index("ix_health_equipment_time", "equipment_id", "at"),)


class MaintenanceRecommendation(Base):
    __tablename__ = "maintenance_recommendations"
    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    site_id: Mapped[str] = mapped_column(ID)
    equipment_id: Mapped[str] = mapped_column(ID, index=True)
    category: Mapped[str] = mapped_column(String(32))
    condition_key: Mapped[str] = mapped_column(String(96))  # e.g. trend:battery_temperature:increasing
    condition: Mapped[str] = mapped_column(Text)
    metric: Mapped[str | None] = mapped_column(String(48))
    action: Mapped[str] = mapped_column(Text)
    evidence_level: Mapped[str] = mapped_column(String(16))  # low | medium | high
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), index=True)  # active | resolved
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_by: Mapped[str | None] = mapped_column(String(100))
    acknowledged_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    data_source: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class AnomalyModel(Base):
    """Metadata of a trained anomaly model (the artifact lives in backend/models/anomaly/)."""

    __tablename__ = "anomaly_models"
    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(32), index=True)
    data_source: Mapped[str] = mapped_column(String(16))  # the model only scores equipment of this source
    algorithm: Mapped[str] = mapped_column(String(32))
    version: Mapped[str] = mapped_column(String(32))
    active: Mapped[bool] = mapped_column(Boolean, default=False)
    artifact_path: Mapped[str] = mapped_column(String(300))
    trained_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    config_hash: Mapped[str] = mapped_column(String(32))
    features: Mapped[list] = mapped_column(JSON, default=list)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    thresholds: Mapped[dict] = mapped_column(JSON, default=dict)
    training: Mapped[dict] = mapped_column(JSON, default=dict)  # window, samples, equipment, data source
    evaluation: Mapped[dict] = mapped_column(JSON, default=dict)


class AnomalyConfig(Base):
    """Versioned detection configuration (YAML text). Exactly one version is active."""

    __tablename__ = "anomaly_configs"
    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    version: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    config_hash: Mapped[str] = mapped_column(String(32))
    active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    created_by: Mapped[str | None] = mapped_column(String(100))
    note: Mapped[str | None] = mapped_column(Text)
