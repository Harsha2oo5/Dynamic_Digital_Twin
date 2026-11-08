"""Telemetry ingestion pipeline: every adapter (HTTP, MQTT, simulator) calls `IngestionPipeline.ingest`.

Per message:
  1. registry     the sensor must be registered and enabled; unregistered sensors are rejected and logged, never
                  auto-created (they cannot touch equipment records)
  2. channel      a sensor only accepts data from its registered protocol; simulated sensors only from the simulator,
                  physical sensors never from the simulator
  3. duplicates   a repeated message_id (per sensor) is dropped; a repeated (sensor, metric, timestamp) is dropped
  4. clock        timestamps more than `max_future_s` ahead of the server are rejected
Per reading:
  5. metric       must be one of the sensor's configured metrics
  6. value        null -> quality "missing"; unit conversion to the canonical unit; implausible -> "invalid"
                  (value stored as NULL, raw value kept for diagnosis)
  7. order        older than the latest reading of that sensor+metric -> "out_of_order": stored for history, never
                  becomes the current value and never raises alerts
Then: store, update the latest-value cache and the sensor's last-seen time, evaluate alert rules, publish.
"""

import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.telemetry import catalog
from app.telemetry.catalog import METRICS, UnitError
from app.telemetry.schemas import IngestMessage, IngestResult

logger = logging.getLogger(__name__)


@dataclass
class SensorInfo:
    sensor_id: str
    equipment_id: str | None
    category: str
    protocol: str
    data_source: str
    enabled: bool
    metrics: frozenset[str]
    sampling_interval_s: float
    stale_after_s: float
    offline_after_s: float
    last_seen_at: datetime | None

    @classmethod
    def from_row(cls, s) -> "SensorInfo":
        stale = s.stale_after_s or max(3 * s.sampling_interval_s, s.sampling_interval_s + 5)
        offline = s.offline_after_s or max(10 * s.sampling_interval_s, 60.0)
        return cls(s.sensor_id, s.equipment_id, s.category, s.protocol, s.data_source, s.enabled,
                   frozenset(s.metrics or []), s.sampling_interval_s, stale, max(offline, stale), s.last_seen_at)


@dataclass
class Latest:
    sensor_id: str
    equipment_id: str | None
    metric: str
    value: float | None
    unit: str
    ts: datetime
    data_source: str
    quality: str
    quality_note: str | None


class _LRU:
    def __init__(self, size: int):
        self.size, self.d = size, OrderedDict()

    def seen(self, key) -> bool:
        if key in self.d:
            self.d.move_to_end(key)
            return True
        self.d[key] = None
        if len(self.d) > self.size:
            self.d.popitem(last=False)
        return False


class IngestionPipeline:
    def __init__(self, store, alerts, publish=lambda kind, payload: None, site_id: str = "SITE-001",
                 max_future_s: float = 60.0):
        self.store, self.alerts, self.publish, self.site_id = store, alerts, publish, site_id
        self.max_future_s = max_future_s
        self._lock = threading.RLock()
        self.sensors: dict[str, SensorInfo] = {}
        self.latest: dict[tuple[str, str], Latest] = {}
        self._message_ids = _LRU(20_000)
        self.stats = {"messages": 0, "stored": 0, "rejected": 0, "duplicates": 0, "invalid": 0, "missing": 0,
                      "out_of_order": 0}
        self._recent = deque(maxlen=5000)  # (monotonic time, readings stored) for the activity rate
        self.recent_errors: deque[dict] = deque(maxlen=50)
        self._dirty_seen: dict[str, datetime] = {}

    # --- registry cache ----------------------------------------------------------------------------
    def load(self) -> None:
        with self._lock:
            self.sensors = {s.sensor_id: SensorInfo.from_row(s) for s in self.store.list_sensors()}
            for m in self.store.latest_measurements():
                self.latest[(m.sensor_id, m.metric)] = Latest(m.sensor_id, m.equipment_id, m.metric, m.value, m.unit,
                                                              m.ts, m.data_source, m.quality, m.quality_note)

    def refresh_sensor(self, sensor_id: str) -> None:
        row = self.store.get_sensor(sensor_id)
        with self._lock:
            if row is None:
                self.sensors.pop(sensor_id, None)
                for key in [k for k in self.latest if k[0] == sensor_id]:
                    del self.latest[key]
            else:
                prev = self.sensors.get(sensor_id)
                info = SensorInfo.from_row(row)
                if prev is not None and prev.last_seen_at and (not info.last_seen_at or prev.last_seen_at > info.last_seen_at):
                    info.last_seen_at = prev.last_seen_at
                self.sensors[sensor_id] = info

    # --- ingestion -----------------------------------------------------------------------------------
    def _reject(self, result: IngestResult, msg: str, sensor_id: str, protocol: str) -> None:
        result.rejected += 1
        result.errors.append(msg)
        self.recent_errors.append({"at": datetime.now(UTC).isoformat(timespec="seconds"), "sensor_id": sensor_id,
                                   "protocol": protocol, "error": msg})
        logger.warning("Telemetry rejected (%s, %s): %s", protocol, sensor_id, msg)

    def ingest(self, message: IngestMessage, protocol: str, received_at: datetime | None = None) -> IngestResult:
        return self.ingest_many([message], protocol, received_at)

    def _check(self, message: IngestMessage, protocol: str, received_at: datetime,
               result: IngestResult) -> tuple[SensorInfo, datetime] | None:
        """Message-level checks; returns (sensor, measurement time) or None when the message is dropped."""
        info = self.sensors.get(message.sensor_id)
        reason = None
        if info is None:
            reason = f"sensor {message.sensor_id} is not registered"
        elif not info.enabled:
            reason = f"sensor {message.sensor_id} is disabled"
        elif (protocol == "simulator") != (info.data_source == "simulated") or (
                protocol != "simulator" and info.protocol != protocol):
            reason = (f"sensor {message.sensor_id} is registered as {info.data_source}/{info.protocol} and does not "
                      f"accept data via {protocol}")
        if reason:
            self._reject(result, reason, message.sensor_id, protocol)
            self.stats["rejected"] += 1
            return None
        if message.message_id and self._message_ids.seen((message.sensor_id, message.message_id)):
            result.duplicates += len(message.readings)
            self.stats["duplicates"] += len(message.readings)
            return None
        ts = message.timestamp.astimezone(UTC) if message.timestamp else received_at
        if ts > received_at + timedelta(seconds=self.max_future_s):
            self._reject(result, f"timestamp {ts.isoformat()} is in the future (sensor clock?)", message.sensor_id,
                         protocol)
            self.stats["rejected"] += 1
            return None
        return info, ts

    def _row(self, info: SensorInfo, message: IngestMessage, r, ts: datetime, received_at: datetime,
             protocol: str) -> dict:
        row = {"sensor_id": info.sensor_id, "equipment_id": info.equipment_id, "site_id": self.site_id,
               "metric": r.metric, "ts": ts, "received_at": received_at, "data_source": info.data_source,
               "protocol": protocol, "message_id": message.message_id, "unit": METRICS[r.metric].unit,
               "value": None, "quality": "good", "quality_note": None, "raw_value": None, "raw_unit": r.unit}
        if r.value is None:
            row["quality"], row["quality_note"] = "missing", "sensor reported no value"
            return row
        row["raw_value"] = repr(r.value)[:64]
        try:
            value, unit = catalog.normalise(r.metric, float(r.value), r.unit)
            row["unit"] = unit
            if not catalog.plausible(r.metric, value):
                row["quality"], row["quality_note"] = "invalid", f"implausible value {value:g} {unit}"
            else:
                row["value"] = round(value, 6)
        except UnitError as exc:
            row["quality"], row["quality_note"] = "invalid", str(exc)[:200]
        return row

    def ingest_many(self, messages: list[IngestMessage], protocol: str,
                    received_at: datetime | None = None) -> IngestResult:
        """Validate every message, then store all accepted readings in ONE transaction (fast for batches and
        backfills); the latest-value cache, alerts and live updates follow in message order."""
        result = IngestResult()
        alerts_todo, published = [], []
        with self._lock:
            rows: list[dict] = []
            late: set[int] = set()
            view: dict[tuple[str, str], datetime] = {}  # latest ts per (sensor, metric) including this batch
            for message in messages:
                rcv = received_at or datetime.now(UTC)
                result.messages += 1
                self.stats["messages"] += 1
                checked = self._check(message, protocol, rcv, result)
                if checked is None:
                    continue
                info, ts = checked
                for r in message.readings:
                    if r.metric not in info.metrics:
                        self._reject(result, f"metric {r.metric!r} is not configured for sensor {info.sensor_id}",
                                     info.sensor_id, protocol)
                        continue
                    key = (info.sensor_id, r.metric)
                    prev = view.get(key) or (self.latest[key].ts if key in self.latest else None)
                    if prev is not None and ts == prev:
                        result.duplicates += 1
                        self.stats["duplicates"] += 1
                        continue
                    row = self._row(info, message, r, ts, rcv, protocol)
                    if prev is not None and ts < prev:
                        note = f"older than latest reading ({prev.isoformat()})"
                        row["quality_note"] = f"{row['quality_note']}; {note}" if row["quality_note"] else note
                        if row["quality"] == "good":
                            row["quality"] = "out_of_order"
                        late.add(id(row))
                    else:
                        view[key] = ts
                    rows.append(row)

            stored, dupes = self.store.insert_measurements(rows)
            dupe_ids = {id(d) for d in dupes}
            result.duplicates += len(dupes)
            self.stats["duplicates"] += len(dupes)
            result.stored += stored
            self.stats["stored"] += stored
            fresh = datetime.now(UTC) - timedelta(seconds=60)  # back-dated history does not count as activity
            self._recent.append((time.monotonic(), sum(1 for r in rows if r["received_at"] >= fresh
                                                       and id(r) not in dupe_ids)))
            for row in rows:
                if id(row) in dupe_ids:
                    continue
                q = row["quality"]  # good | missing | invalid | out_of_order (all IngestResult counters)
                setattr(result, q, getattr(result, q) + 1)
                if q in self.stats:
                    self.stats[q] += 1
                info = self.sensors[row["sensor_id"]]
                if info.last_seen_at is None or row["received_at"] > info.last_seen_at:
                    info.last_seen_at = row["received_at"]
                    self._dirty_seen[info.sensor_id] = row["received_at"]
                if id(row) in late:
                    continue  # history only: never the current value, never an alert
                latest = Latest(row["sensor_id"], row["equipment_id"], row["metric"], row["value"], row["unit"],
                                row["ts"], row["data_source"], q, row["quality_note"])
                self.latest[(row["sensor_id"], row["metric"])] = latest
                published.append(latest)
                if q == "good":
                    alerts_todo.append((info, row))
        result.errors = result.errors[:50]
        if self.alerts is not None:
            for info, row in alerts_todo:
                self.alerts.evaluate(info.sensor_id, info.equipment_id, info.category, row["metric"], row["value"],
                                     row["ts"], info.data_source)
        if published:
            self.publish("measurements", published)
        return result

    def flush_last_seen(self) -> None:
        """Persist last-communication times (batched; called by the watchdog and at shutdown)."""
        with self._lock:
            dirty, self._dirty_seen = self._dirty_seen, {}
        self.store.touch_sensors(dirty)

    def readings_per_minute(self) -> float:
        cutoff = time.monotonic() - 60
        return float(sum(n for t, n in self._recent if t >= cutoff))
