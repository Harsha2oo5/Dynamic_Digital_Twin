"""Pydantic contract of the telemetry API (requests, responses and WebSocket payloads)."""

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.telemetry.catalog import EQUIPMENT_CATEGORIES, METRICS, PROTOCOLS, SENSOR_TYPES

ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$"
SECRET_KEY = re.compile(r"(pass(word)?|secret|token|api[_-]?key|credential|private[_-]?key)", re.I)

SensorStatus = Literal["unknown", "online", "stale", "offline", "disabled"]
DataSource = Literal["physical", "simulated"]
Quality = Literal["good", "missing", "invalid", "out_of_order"]


def _check_config(config: dict) -> dict:
    """Credentials never live in the database or API responses; reference an environment variable instead."""
    def walk(d, path=""):
        for k, v in d.items():
            if SECRET_KEY.search(str(k)) and not str(k).endswith("_env"):
                raise ValueError(f"config key {path + str(k)!r} looks like a secret; store it in an environment "
                                 f"variable and reference it as '{k}_env'")
            if isinstance(v, dict):
                walk(v, f"{path}{k}.")
    walk(config)
    return config


def check_sensor_type(v: str) -> str:
    if v not in SENSOR_TYPES:
        raise ValueError(f"sensor_type must be one of {SENSOR_TYPES}")
    return v


def check_protocol(v: str) -> str:
    if v not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    return v


def check_metrics(v: list[str]) -> list[str]:
    unknown = [m for m in v if m not in METRICS]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; see GET /api/telemetry/metrics")
    if not v:
        raise ValueError("at least one metric is required")
    return list(dict.fromkeys(v))


# --- sites / equipment ----------------------------------------------------------------------------
class SiteOut(BaseModel):
    site_id: str
    name: str
    location: str | None = None


class EquipmentCreate(BaseModel):
    equipment_id: str = Field(pattern=ID_PATTERN, description="Stable operator-assigned ID, e.g. BAT-001")
    category: Literal["antenna", "router", "cable", "battery", "power_supply_unit", "equipment_cabinet", "radio_unit"]
    name: str = Field(min_length=1, max_length=200)
    zone: str | None = Field(None, max_length=100)
    description: str | None = None
    data_source: DataSource = "physical"


class EquipmentUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    zone: str | None = Field(None, max_length=100)
    description: str | None = None


class EquipmentOut(BaseModel):
    equipment_id: str
    site_id: str
    category: str
    name: str
    zone: str | None
    description: str | None
    data_source: DataSource
    sensor_ids: list[str] = []
    sensors_online: int = 0
    last_updated: datetime | None = None
    active_alerts: int = 0
    visual_links: list[dict] = []


class VisualLinkIn(BaseModel):
    object_id: str = Field(min_length=1, max_length=128, description="Inventory object id (source:track_id)")
    class_name: str | None = None
    linked_by: str | None = Field(None, max_length=100)
    note: str | None = None


# --- sensors ----------------------------------------------------------------------------------------
class SensorBase(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    sensor_type: str
    protocol: str
    metrics: list[str] = Field(min_length=1)
    zone: str | None = Field(None, max_length=100)
    sampling_interval_s: float = Field(10.0, gt=0, le=86400)
    stale_after_s: float | None = Field(None, gt=0)
    offline_after_s: float | None = Field(None, gt=0)
    config: dict = {}
    enabled: bool = True

    @field_validator("sensor_type")
    @classmethod
    def _type(cls, v):
        return check_sensor_type(v)

    @field_validator("protocol")
    @classmethod
    def _protocol(cls, v):
        return check_protocol(v)

    @field_validator("metrics")
    @classmethod
    def _metrics(cls, v):
        return check_metrics(v)

    @field_validator("config")
    @classmethod
    def _config(cls, v):
        return _check_config(v)


class SensorCreate(SensorBase):
    sensor_id: str = Field(pattern=ID_PATTERN)
    equipment_id: str | None = Field(None, pattern=ID_PATTERN)
    category: str | None = Field(None, description="Equipment category or 'site'; taken from the equipment if linked")
    data_source: DataSource = "physical"

    @model_validator(mode="after")
    def _category(self):
        if self.category is not None and self.category not in EQUIPMENT_CATEGORIES:
            raise ValueError(f"category must be one of {EQUIPMENT_CATEGORIES}")
        if self.equipment_id is None and self.category is None:
            raise ValueError("give equipment_id or category ('site' for site-level sensors)")
        return self


class SensorUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    sensor_type: str | None = None
    metrics: list[str] | None = None
    zone: str | None = None
    sampling_interval_s: float | None = Field(None, gt=0, le=86400)
    stale_after_s: float | None = Field(None, gt=0)
    offline_after_s: float | None = Field(None, gt=0)
    config: dict | None = None
    enabled: bool | None = None

    @field_validator("sensor_type")
    @classmethod
    def _type(cls, v):
        return v if v is None else check_sensor_type(v)

    @field_validator("metrics")
    @classmethod
    def _metrics(cls, v):
        return v if v is None else check_metrics(v)

    @field_validator("config")
    @classmethod
    def _config(cls, v):
        return v if v is None else _check_config(v)


class AssociationIn(BaseModel):
    equipment_id: str | None = Field(None, pattern=ID_PATTERN, description="null removes the association")


class SensorOut(BaseModel):
    sensor_id: str
    name: str
    sensor_type: str
    protocol: str
    site_id: str
    equipment_id: str | None
    category: str
    zone: str | None
    data_source: DataSource
    enabled: bool
    metrics: list[str]
    sampling_interval_s: float
    stale_after_s: float
    offline_after_s: float
    config: dict
    status: SensorStatus
    last_seen_at: datetime | None
    seconds_since_seen: float | None


# --- ingestion -----------------------------------------------------------------------------------
class IngestReading(BaseModel):
    metric: str = Field(min_length=1, max_length=48)
    value: float | int | bool | None = Field(description="null = the sensor reports the value as unavailable")
    unit: str | None = Field(None, max_length=16)


class IngestMessage(BaseModel):
    sensor_id: str = Field(pattern=ID_PATTERN)
    message_id: str | None = Field(None, max_length=128, description="Unique per sensor; enables duplicate detection")
    timestamp: datetime | None = Field(None, description="Measurement time, ISO-8601 with offset; server time if absent")
    readings: list[IngestReading] = Field(min_length=1, max_length=100)

    @field_validator("timestamp")
    @classmethod
    def _aware(cls, v):
        if v is not None and v.tzinfo is None:
            raise ValueError("timestamp needs a timezone offset (e.g. 2026-11-07T10:00:00Z)")
        return v


class IngestBatch(BaseModel):
    messages: list[IngestMessage] = Field(min_length=1, max_length=500)


class IngestResult(BaseModel):
    messages: int = 0
    stored: int = 0
    good: int = 0
    missing: int = 0
    invalid: int = 0
    out_of_order: int = 0
    duplicates: int = 0
    rejected: int = 0
    errors: list[str] = []


# --- measurements -------------------------------------------------------------------------------
class MeasurementOut(BaseModel):
    sensor_id: str
    equipment_id: str | None
    metric: str
    value: float | None
    unit: str
    ts: datetime
    data_source: DataSource
    quality: Quality
    quality_note: str | None = None
    state_label: str | None = None


class LatestReading(MeasurementOut):
    label: str
    sensor_status: SensorStatus
    age_s: float
    current: bool = Field(description="False when the sensor is stale/offline: show as last known value, not current")


class HistoryPoint(BaseModel):
    t: datetime
    avg: float
    min: float
    max: float
    n: int


class HistorySeries(BaseModel):
    sensor_id: str
    equipment_id: str | None
    metric: str
    label: str
    unit: str
    data_source: DataSource
    bucket_s: int
    points: list[HistoryPoint]


class Page(BaseModel):
    total: int
    limit: int
    offset: int
    items: list


# --- alerts ---------------------------------------------------------------------------------------
class AlertOut(BaseModel):
    id: int
    rule_id: str
    severity: Literal["warning", "critical"]
    state: Literal["active", "cleared"]
    sensor_id: str | None
    equipment_id: str | None
    metric: str | None
    value: float | None
    peak_value: float | None
    threshold: float | None
    message: str
    data_source: DataSource
    opened_at: datetime
    last_triggered_at: datetime
    cleared_at: datetime | None
    acknowledged: bool
    acknowledged_at: datetime | None
    acknowledged_by: str | None
    ack_note: str | None


class AlertAck(BaseModel):
    acknowledged_by: str = Field(min_length=1, max_length=100)
    note: str | None = Field(None, max_length=1000)


# --- simulator ------------------------------------------------------------------------------------
class SimulatorStart(BaseModel):
    interval_scale: float = Field(1.0, gt=0, le=100, description="divides every sensor's sampling interval")
    backfill_hours: float = Field(0.0, ge=0, le=24 * 7, description="generate this much clearly-simulated history first")
    seed: int | None = None


class SimulatorScenario(BaseModel):
    scenario: Literal["mains_outage", "sensor_dropout", "door_open", "fan_failure", "heat_wave", "temperature_spike",
                      "power_anomaly", "voltage_fluctuation", "battery_degradation", "psu_overheating"]
    duration_s: float = Field(300, gt=0, le=14 * 86400)
    sensor_id: str | None = Field(None, description="sensor_dropout: which simulated sensor (default: a random one)")
