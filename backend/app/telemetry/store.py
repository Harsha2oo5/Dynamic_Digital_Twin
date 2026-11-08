"""Database access for the telemetry subsystem (synchronous SQLAlchemy; called from worker threads)."""

from datetime import datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.telemetry.models import (
    Alert,
    Equipment,
    EquipmentVisualLink,
    Measurement,
    Sensor,
    SensorStatusEvent,
    Site,
    utcnow,
)


class NotFound(LookupError):
    pass


class Conflict(ValueError):
    pass


class TelemetryStore:
    def __init__(self, sessions: sessionmaker[Session]):
        self.sessions = sessions

    # --- sites ------------------------------------------------------------------------------------
    def ensure_site(self, site_id: str, name: str) -> Site:
        with self.sessions() as s:
            site = s.get(Site, site_id)
            if site is None:
                site = Site(site_id=site_id, name=name)
                s.add(site)
                s.commit()
            return site

    # --- equipment --------------------------------------------------------------------------------
    def create_equipment(self, site_id: str, data: dict) -> Equipment:
        with self.sessions() as s:
            if s.get(Equipment, data["equipment_id"]) is not None:
                raise Conflict(f"equipment {data['equipment_id']} already exists")
            eq = Equipment(site_id=site_id, **data)
            s.add(eq)
            s.commit()
            return eq

    def update_equipment(self, equipment_id: str, changes: dict) -> Equipment:
        with self.sessions() as s:
            eq = s.get(Equipment, equipment_id)
            if eq is None:
                raise NotFound(f"equipment {equipment_id} not found")
            for k, v in changes.items():
                setattr(eq, k, v)
            s.commit()
            return eq

    def delete_equipment(self, equipment_id: str) -> None:
        with self.sessions() as s:
            eq = s.get(Equipment, equipment_id)
            if eq is None:
                raise NotFound(f"equipment {equipment_id} not found")
            for sensor in s.scalars(select(Sensor).where(Sensor.equipment_id == equipment_id)):
                sensor.equipment_id = None  # sensors stay registered; history keeps its equipment snapshot
            s.delete(eq)
            s.commit()

    def get_equipment(self, equipment_id: str) -> Equipment | None:
        with self.sessions() as s:
            return s.get(Equipment, equipment_id)

    def list_equipment(self, category: str | None = None, data_source: str | None = None) -> list[Equipment]:
        with self.sessions() as s:
            q = select(Equipment).order_by(Equipment.category, Equipment.equipment_id)
            if category:
                q = q.where(Equipment.category == category)
            if data_source:
                q = q.where(Equipment.data_source == data_source)
            return list(s.scalars(q))

    def add_visual_link(self, equipment_id: str, data: dict) -> EquipmentVisualLink:
        with self.sessions() as s:
            if s.get(Equipment, equipment_id) is None:
                raise NotFound(f"equipment {equipment_id} not found")
            link = EquipmentVisualLink(equipment_id=equipment_id, **data)
            s.add(link)
            s.commit()
            return link

    def visual_links(self, equipment_id: str | None = None) -> list[EquipmentVisualLink]:
        with self.sessions() as s:
            q = select(EquipmentVisualLink).order_by(EquipmentVisualLink.linked_at)
            if equipment_id:
                q = q.where(EquipmentVisualLink.equipment_id == equipment_id)
            return list(s.scalars(q))

    def delete_visual_link(self, link_id: int) -> None:
        with self.sessions() as s:
            link = s.get(EquipmentVisualLink, link_id)
            if link is None:
                raise NotFound(f"link {link_id} not found")
            s.delete(link)
            s.commit()

    # --- sensors ----------------------------------------------------------------------------------
    def create_sensor(self, site_id: str, data: dict) -> Sensor:
        with self.sessions() as s:
            if s.get(Sensor, data["sensor_id"]) is not None:
                raise Conflict(f"sensor {data['sensor_id']} already exists")
            if data.get("equipment_id"):
                eq = s.get(Equipment, data["equipment_id"])
                if eq is None:
                    raise NotFound(f"equipment {data['equipment_id']} not found")
                if data.get("category") not in (None, eq.category):
                    raise Conflict(f"sensor category {data['category']} differs from equipment category {eq.category}")
                data["category"] = eq.category
            sensor = Sensor(site_id=site_id, **data)
            s.add(sensor)
            s.commit()
            return sensor

    def update_sensor(self, sensor_id: str, changes: dict) -> Sensor:
        with self.sessions() as s:
            sensor = s.get(Sensor, sensor_id)
            if sensor is None:
                raise NotFound(f"sensor {sensor_id} not found")
            for k, v in changes.items():
                setattr(sensor, k, v)
            s.commit()
            return sensor

    def associate(self, sensor_id: str, equipment_id: str | None) -> Sensor:
        with self.sessions() as s:
            sensor = s.get(Sensor, sensor_id)
            if sensor is None:
                raise NotFound(f"sensor {sensor_id} not found")
            if equipment_id is not None:
                eq = s.get(Equipment, equipment_id)
                if eq is None:
                    raise NotFound(f"equipment {equipment_id} not found")
                if eq.data_source != sensor.data_source:
                    raise Conflict(f"cannot attach a {sensor.data_source} sensor to {eq.data_source} equipment")
                sensor.category = eq.category
            sensor.equipment_id = equipment_id
            s.commit()
            return sensor

    def delete_sensor(self, sensor_id: str) -> None:
        with self.sessions() as s:
            sensor = s.get(Sensor, sensor_id)
            if sensor is None:
                raise NotFound(f"sensor {sensor_id} not found")
            s.delete(sensor)
            s.commit()

    def get_sensor(self, sensor_id: str) -> Sensor | None:
        with self.sessions() as s:
            return s.get(Sensor, sensor_id)

    def list_sensors(self) -> list[Sensor]:
        with self.sessions() as s:
            return list(s.scalars(select(Sensor).order_by(Sensor.sensor_id)))

    def touch_sensors(self, last_seen: dict[str, datetime]) -> None:
        if not last_seen:
            return
        with self.sessions() as s:
            for sid, at in last_seen.items():
                sensor = s.get(Sensor, sid)
                if sensor is not None and (sensor.last_seen_at is None or at > sensor.last_seen_at):
                    sensor.last_seen_at = at
            s.commit()

    def add_status_events(self, events: list[tuple[str, str, datetime]]) -> None:
        if not events:
            return
        with self.sessions() as s:
            s.add_all(SensorStatusEvent(sensor_id=a, status=b, at=c) for a, b, c in events)
            s.commit()

    def status_events(self, since: datetime, sensor_id: str | None = None) -> list[SensorStatusEvent]:
        with self.sessions() as s:
            q = select(SensorStatusEvent).where(SensorStatusEvent.at >= since).order_by(SensorStatusEvent.at)
            if sensor_id:
                q = q.where(SensorStatusEvent.sensor_id == sensor_id)
            return list(s.scalars(q))

    def last_status_before(self, before: datetime) -> dict[str, str]:
        """Each sensor's status at `before` (its latest event before that time)."""
        with self.sessions() as s:
            sub = (select(SensorStatusEvent.sensor_id, func.max(SensorStatusEvent.at).label("at"))
                   .where(SensorStatusEvent.at < before).group_by(SensorStatusEvent.sensor_id).subquery())
            q = select(SensorStatusEvent.sensor_id, SensorStatusEvent.status).join(
                sub, (SensorStatusEvent.sensor_id == sub.c.sensor_id) & (SensorStatusEvent.at == sub.c.at))
            return {a: b for a, b in s.execute(q)}

    # --- measurements -----------------------------------------------------------------------------
    def insert_measurements(self, rows: list[dict]) -> tuple[int, list[dict]]:
        """Insert rows; rows that violate (sensor, metric, ts) uniqueness are duplicates. Returns (stored, dupes)."""
        if not rows:
            return 0, []
        with self.sessions() as s:
            try:
                s.add_all(Measurement(**r) for r in rows)
                s.commit()
                return len(rows), []
            except IntegrityError:
                s.rollback()
        stored, dupes = 0, []
        with self.sessions() as s:
            for r in rows:
                try:
                    with s.begin_nested():
                        s.add(Measurement(**r))
                    stored += 1
                except IntegrityError:
                    dupes.append(r)
            s.commit()
        return stored, dupes

    def latest_measurements(self) -> list[Measurement]:
        """Latest in-order row per (sensor, metric), used to warm the cache at startup."""
        with self.sessions() as s:
            sub = (select(Measurement.sensor_id, Measurement.metric, func.max(Measurement.ts).label("ts"))
                   .where(Measurement.quality != "out_of_order")
                   .group_by(Measurement.sensor_id, Measurement.metric).subquery())
            q = select(Measurement).join(sub, (Measurement.sensor_id == sub.c.sensor_id)
                                         & (Measurement.metric == sub.c.metric) & (Measurement.ts == sub.c.ts))
            return list(s.scalars(q))

    def series(self, metric: str, start: datetime, end: datetime, sensor_id: str | None = None,
               equipment_id: str | None = None, limit: int = 200_000) -> list[tuple[str, str | None, datetime, float, str]]:
        """(sensor_id, equipment_id, ts, value, data_source) of good readings, oldest first."""
        with self.sessions() as s:
            q = (select(Measurement.sensor_id, Measurement.equipment_id, Measurement.ts, Measurement.value,
                        Measurement.data_source)
                 .where(Measurement.metric == metric, Measurement.ts >= start, Measurement.ts <= end,
                        Measurement.quality == "good", Measurement.value.is_not(None))
                 .order_by(Measurement.ts).limit(limit))
            if sensor_id:
                q = q.where(Measurement.sensor_id == sensor_id)
            if equipment_id:
                q = q.where(Measurement.equipment_id == equipment_id)
            return [tuple(r) for r in s.execute(q)]

    def measurements_page(self, limit: int, offset: int, **filters) -> tuple[int, list[Measurement]]:
        with self.sessions() as s:
            q = select(Measurement)
            for col in ("sensor_id", "equipment_id", "metric", "quality", "data_source"):
                if filters.get(col):
                    q = q.where(getattr(Measurement, col) == filters[col])
            if filters.get("start"):
                q = q.where(Measurement.ts >= filters["start"])
            if filters.get("end"):
                q = q.where(Measurement.ts <= filters["end"])
            total = s.scalar(select(func.count()).select_from(q.subquery()))
            rows = list(s.scalars(q.order_by(Measurement.ts.desc(), Measurement.id.desc()).limit(limit).offset(offset)))
            return total, rows

    def counts(self) -> dict:
        with self.sessions() as s:
            return {"measurements": s.scalar(select(func.count(Measurement.id))),
                    "oldest": s.scalar(select(func.min(Measurement.ts)))}

    def purge(self, retention_days: int) -> int:
        cutoff = utcnow() - timedelta(days=retention_days)
        with self.sessions() as s:
            n = s.execute(delete(Measurement).where(Measurement.ts < cutoff)).rowcount
            s.execute(delete(SensorStatusEvent).where(SensorStatusEvent.at < cutoff))
            s.execute(delete(Alert).where(Alert.state == "cleared", Alert.cleared_at < cutoff))
            s.commit()
            return n or 0

    # --- alerts -----------------------------------------------------------------------------------
    def active_alerts(self) -> list[Alert]:
        with self.sessions() as s:
            return list(s.scalars(select(Alert).where(Alert.state == "active").order_by(Alert.opened_at)))

    def save_alert(self, alert: Alert) -> Alert:
        with self.sessions() as s:
            merged = s.merge(alert)
            s.commit()
            return merged

    def get_alert(self, alert_id: int) -> Alert | None:
        with self.sessions() as s:
            return s.get(Alert, alert_id)

    def alerts_page(self, limit: int, offset: int, state: str | None = None, severity: str | None = None,
                    equipment_id: str | None = None, sensor_id: str | None = None,
                    acknowledged: bool | None = None) -> tuple[int, list[Alert]]:
        with self.sessions() as s:
            q = select(Alert)
            if state:
                q = q.where(Alert.state == state)
            if severity:
                q = q.where(Alert.severity == severity)
            if equipment_id:
                q = q.where(Alert.equipment_id == equipment_id)
            if sensor_id:
                q = q.where(Alert.sensor_id == sensor_id)
            if acknowledged is not None:
                q = q.where(Alert.acknowledged == acknowledged)
            total = s.scalar(select(func.count()).select_from(q.subquery()))
            rows = list(s.scalars(q.order_by(Alert.opened_at.desc()).limit(limit).offset(offset)))
            return total, rows

    def acknowledge(self, alert_id: int, by: str, note: str | None) -> Alert:
        with self.sessions() as s:
            a = s.get(Alert, alert_id)
            if a is None:
                raise NotFound(f"alert {alert_id} not found")
            a.acknowledged, a.acknowledged_at, a.acknowledged_by, a.ack_note = True, utcnow(), by, note
            s.commit()
            return a
