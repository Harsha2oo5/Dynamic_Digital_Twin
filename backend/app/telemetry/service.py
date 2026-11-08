"""TelemetryService: owns the database, ingestion pipeline, alert engine, hub, adapters, simulator and the
connection watchdog. Created at startup and attached to app.state.telemetry.

If the database cannot be reached the service is marked unavailable (telemetry endpoints answer 503 with the
reason) and the rest of the application keeps working.
"""

import logging
import threading
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta

from app.core.config import Settings
from app.telemetry import db
from app.telemetry.adapters import MqttAdapter
from app.telemetry.alerts import COMMUNICATION_RULE, AlertEngine, load_rules
from app.telemetry.catalog import METRICS
from app.telemetry.hub import TelemetryHub
from app.telemetry.ingest import IngestionPipeline, Latest, SensorInfo
from app.telemetry.simulator import SIMULATOR_PROTOCOL, Simulator
from app.telemetry.store import TelemetryStore

logger = logging.getLogger(__name__)

RANGES = {"15m": 900, "1h": 3600, "6h": 6 * 3600, "24h": 86400, "7d": 7 * 86400}


def sensor_status(info: SensorInfo, now: datetime) -> str:
    if not info.enabled:
        return "disabled"
    if info.last_seen_at is None:
        return "unknown"
    age = (now - info.last_seen_at).total_seconds()
    if age <= info.stale_after_s:
        return "online"
    if age <= info.offline_after_s:
        return "stale"
    return "offline"


def bucket_series(rows: list[tuple], start: datetime, end: datetime, max_points: int) -> tuple[int, list[dict]]:
    """Average / min / max per time bucket so a chart never receives more than `max_points` points."""
    span = max(1.0, (end - start).total_seconds())
    bucket = max(1, int(-(-span // max_points)))
    acc: dict[int, list[float]] = defaultdict(list)
    for *_, ts, value, _src in rows:
        acc[int((ts - start).total_seconds() // bucket)].append(value)
    points = [{"t": start + timedelta(seconds=k * bucket), "avg": round(sum(v) / len(v), 4), "min": round(min(v), 4),
               "max": round(max(v), 4), "n": len(v)} for k, v in sorted(acc.items())]
    return bucket, points


class TelemetryUnavailable(RuntimeError):
    pass


class TelemetryService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.site_id = settings.site_id
        self.hub = TelemetryHub()
        self.error: str | None = None
        self.db_kind: str | None = None
        self.engine = None
        self.store: TelemetryStore | None = None
        self.pipeline: IngestionPipeline | None = None
        self.alerts: AlertEngine | None = None
        self.simulator: Simulator | None = None
        self.mqtt: MqttAdapter | None = None
        self._status: dict[str, tuple[str, datetime]] = {}  # sensor -> (status, since)
        self._stop = threading.Event()
        self._watchdog: threading.Thread | None = None
        self._last_purge: datetime | None = None
        self.started = False
        # called with every batch of new in-order readings (e.g. the anomaly engine); must never raise
        self.listeners: list = []

    # --- lifecycle ------------------------------------------------------------------------------------
    def start(self) -> None:
        s = self.settings
        if not s.telemetry_enabled:
            self.error = "telemetry is disabled (TELEMETRY_ENABLED=false)"
            return
        try:
            url, self.db_kind = db.database_url(s)
            self.engine = db.make_engine(url)
            db.upgrade(self.engine)
            self.store = TelemetryStore(db.session_factory(self.engine))
            self.store.ensure_site(self.site_id, s.site_name)
            self.alerts = AlertEngine(load_rules(s.resolve(s.telemetry_thresholds_file)), self.store, self.site_id,
                                      self.publish)
            self.pipeline = IngestionPipeline(self.store, self.alerts, self.publish, self.site_id)
            self.pipeline.load()
            prior = self.store.last_status_before(datetime.now(UTC) + timedelta(seconds=1))
            now = datetime.now(UTC)
            self._status = {sid: (st, now) for sid, st in prior.items()}
            self.simulator = Simulator(s.resolve(s.simulator_config_file), self.pipeline, self.register_simulated,
                                       s.app_env)
        except Exception as exc:  # noqa: BLE001
            self.error = f"telemetry database unavailable ({self.db_kind or 'unknown'}): {type(exc).__name__}"
            logger.error("Telemetry disabled: %s", exc.__class__.__name__)
            logger.debug("Telemetry start failure", exc_info=True)
            return
        if self.db_kind and self.db_kind.startswith("sqlite"):
            logger.warning("Telemetry uses the SQLite development fallback; set TELEMETRY_DATABASE_URL for PostgreSQL")
        self._watchdog = threading.Thread(target=self._watch, name="telemetry-watchdog", daemon=True)
        self._watchdog.start()
        if s.mqtt_enabled:
            self.mqtt = MqttAdapter(s, self.pipeline)
            self.mqtt.start()
        if s.simulator_autostart and s.app_env != "production":
            self.simulator.start()
        self.started = True
        logger.info("Telemetry started (%s, %d sensors)", self.db_kind, len(self.pipeline.sensors))

    def stop(self) -> None:
        self._stop.set()
        if self.simulator is not None and self.simulator.running:
            self.simulator.stop()
        if self.mqtt is not None:
            self.mqtt.stop()
        if self._watchdog is not None:
            self._watchdog.join(5)
        if self.pipeline is not None:
            self.pipeline.flush_last_seen()
        if self.engine is not None:
            self.engine.dispose()

    def require(self) -> None:
        if not self.started:
            raise TelemetryUnavailable(self.error or "telemetry is starting")

    # --- publishing --------------------------------------------------------------------------------
    def publish(self, kind: str, payload) -> None:
        if kind == "measurements":
            for listener in list(self.listeners):
                try:
                    listener(payload)
                except Exception:  # noqa: BLE001 - analytics problems must not break ingestion
                    logger.exception("Telemetry listener failed")
            event = {"type": "measurements", "items": [self.latest_out(l) for l in payload]}
        elif kind == "alert":
            event = {"type": "alert", "alert": self.alert_out(payload)}
        else:
            event = {"type": kind, **payload}
        self.hub.publish(event)

    # --- watchdog ------------------------------------------------------------------------------------
    def _watch(self) -> None:
        while not self._stop.wait(self.settings.telemetry_watchdog_interval_s):
            try:
                self.pipeline.flush_last_seen()
                self.check_status()
                now = datetime.now(UTC)
                if self._last_purge is None or now - self._last_purge > timedelta(hours=1):
                    self._last_purge = now
                    n = self.store.purge(self.settings.telemetry_retention_days)
                    if n:
                        logger.info("Telemetry retention: purged %d measurements", n)
            except Exception:  # noqa: BLE001
                logger.exception("Telemetry watchdog iteration failed")

    def check_status(self, now: datetime | None = None) -> list[dict]:
        """Recompute every sensor's connection state; record and publish transitions; raise/clear comm alerts."""
        now = now or datetime.now(UTC)
        changes, events = [], []
        # an intentionally stopped simulator, or one still writing back-dated history, is not a communication failure
        sim_running = self.simulator is not None and self.simulator.running and not self.simulator.backfilling
        for info in list(self.pipeline.sensors.values()):
            st = sensor_status(info, now)
            prev = self._status.get(info.sensor_id)
            if prev is None or prev[0] != st:
                since = now if st != "offline" or info.last_seen_at is None else \
                    info.last_seen_at + timedelta(seconds=info.offline_after_s)
                self._status[info.sensor_id] = (st, since)
                events.append((info.sensor_id, st, now))
                changes.append({"sensor_id": info.sensor_id, "status": st, "last_seen_at": info.last_seen_at,
                                "data_source": info.data_source})
            # an intentionally stopped simulator is not a communication failure of the simulated sensors
            if info.data_source == "simulated" and not sim_running:
                continue
            if st in ("offline", "online"):
                self.alerts.sensor_status(info.sensor_id, info.equipment_id, st, self._status[info.sensor_id][1], now,
                                          info.data_source)
        self.store.add_status_events(events)
        for c in changes:
            self.publish("sensor_status", {**c, "last_seen_at": c["last_seen_at"].isoformat() if c["last_seen_at"]
                                           else None})
        return changes

    # --- registry ------------------------------------------------------------------------------------
    def register_simulated(self, cfg: dict) -> None:
        """Create the simulator's equipment and sensors if missing (data_source=simulated, protocol=simulator)."""
        for e in cfg.get("equipment", []):
            if self.store.get_equipment(e["id"]) is None:
                self.store.create_equipment(self.site_id, {"equipment_id": e["id"], "category": e["category"],
                                                           "name": e["name"], "zone": e.get("zone"),
                                                           "data_source": "simulated",
                                                           "description": "Simulated equipment (no physical device)"})
            for sc in e.get("sensors", []):
                self._register_sim_sensor(sc, e["id"], e["category"], e.get("zone"))
        for sc in cfg.get("site_sensors", []):
            self._register_sim_sensor(sc, None, "site", "Site")

    def _register_sim_sensor(self, sc: dict, equipment_id: str | None, category: str, zone: str | None) -> None:
        if self.store.get_sensor(sc["id"]) is None:
            self.store.create_sensor(self.site_id, {
                "sensor_id": sc["id"], "name": f"Simulated {sc['type'].replace('_', ' ')} sensor",
                "sensor_type": sc["type"], "protocol": SIMULATOR_PROTOCOL, "equipment_id": equipment_id,
                "category": category, "zone": zone, "data_source": "simulated", "metrics": sc["metrics"],
                "sampling_interval_s": float(sc["interval_s"]), "config": {"simulated": True}})
        self.pipeline.refresh_sensor(sc["id"])

    def sensor_changed(self, sensor_id: str, deleted: bool = False) -> None:
        self.pipeline.refresh_sensor(sensor_id)
        if deleted:
            self._status.pop(sensor_id, None)
            self.alerts.forget_sensor(sensor_id)

    # --- read models -------------------------------------------------------------------------------
    def sensor_out(self, row, now: datetime | None = None) -> dict:
        now = now or datetime.now(UTC)
        info = self.pipeline.sensors.get(row.sensor_id) or SensorInfo.from_row(row)
        last = info.last_seen_at or row.last_seen_at
        return {"sensor_id": row.sensor_id, "name": row.name, "sensor_type": row.sensor_type, "protocol": row.protocol,
                "site_id": row.site_id, "equipment_id": row.equipment_id, "category": row.category, "zone": row.zone,
                "data_source": row.data_source, "enabled": row.enabled, "metrics": list(row.metrics or []),
                "sampling_interval_s": row.sampling_interval_s, "stale_after_s": info.stale_after_s,
                "offline_after_s": info.offline_after_s, "config": row.config or {},
                "status": sensor_status(info, now), "last_seen_at": last,
                "seconds_since_seen": round((now - last).total_seconds(), 1) if last else None}

    def latest_out(self, l: Latest, now: datetime | None = None) -> dict:
        now = now or datetime.now(UTC)
        info = self.pipeline.sensors.get(l.sensor_id)
        st = sensor_status(info, now) if info else "unknown"
        m = METRICS.get(l.metric)
        state = m.states.get(int(l.value)) if (m and m.states and l.value is not None) else None
        return {"sensor_id": l.sensor_id, "equipment_id": l.equipment_id, "metric": l.metric, "value": l.value,
                "unit": l.unit, "ts": l.ts.isoformat(), "data_source": l.data_source, "quality": l.quality,
                "quality_note": l.quality_note, "state_label": state, "label": m.label if m else l.metric,
                "sensor_status": st, "age_s": round((now - l.ts).total_seconds(), 1),
                "current": st == "online" and l.quality == "good"}

    @staticmethod
    def alert_out(a) -> dict:
        return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in {
            "id": a.id, "rule_id": a.rule_id, "severity": a.severity, "state": a.state, "sensor_id": a.sensor_id,
            "equipment_id": a.equipment_id, "metric": a.metric, "value": a.value, "peak_value": a.peak_value,
            "threshold": a.threshold, "message": a.message, "data_source": a.data_source, "opened_at": a.opened_at,
            "last_triggered_at": a.last_triggered_at, "cleared_at": a.cleared_at, "acknowledged": a.acknowledged,
            "acknowledged_at": a.acknowledged_at, "acknowledged_by": a.acknowledged_by, "ack_note": a.ack_note}.items()}

    def latest(self, sensor_id: str | None = None, equipment_id: str | None = None, metric: str | None = None,
               category: str | None = None) -> list[dict]:
        now = datetime.now(UTC)
        out = []
        for (sid, met), l in sorted(self.pipeline.latest.items()):
            info = self.pipeline.sensors.get(sid)
            if info is None or (sensor_id and sid != sensor_id) or (metric and met != metric) or (
                    equipment_id and info.equipment_id != equipment_id) or (category and info.category != category):
                continue
            out.append(self.latest_out(l, now))
        return out

    def equipment_out(self, eq, now: datetime | None = None) -> dict:
        now = now or datetime.now(UTC)
        sensors = [i for i in self.pipeline.sensors.values() if i.equipment_id == eq.equipment_id]
        seen = [i.last_seen_at for i in sensors if i.last_seen_at]
        active = [a for a in self.alerts.active() if a.equipment_id == eq.equipment_id]
        return {"equipment_id": eq.equipment_id, "site_id": eq.site_id, "category": eq.category, "name": eq.name,
                "zone": eq.zone, "description": eq.description, "data_source": eq.data_source,
                "sensor_ids": sorted(i.sensor_id for i in sensors),
                "sensors_online": sum(sensor_status(i, now) == "online" for i in sensors),
                "last_updated": max(seen) if seen else None, "active_alerts": len(active),
                "visual_links": [{"id": v.id, "object_id": v.object_id, "class_name": v.class_name,
                                  "linked_by": v.linked_by, "note": v.note, "linked_at": v.linked_at}
                                 for v in self.store.visual_links(eq.equipment_id)]}

    def summary(self) -> dict:
        now = datetime.now(UTC)
        infos = list(self.pipeline.sensors.values())
        st = Counter(sensor_status(i, now) for i in infos)
        ambient = [l for (sid, m), l in self.pipeline.latest.items() if m == "ambient_temperature" and l.value is not None]
        ambient.sort(key=lambda l: l.ts, reverse=True)
        cab = [l for (sid, m), l in self.pipeline.latest.items() if m == "internal_temperature" and l.value is not None]
        cab.sort(key=lambda l: l.ts, reverse=True)
        def online_power(categories: tuple[str, ...]) -> list[Latest]:
            return [l for (sid, m), l in self.pipeline.latest.items() if m == "power_consumption"
                    and l.value is not None and l.quality == "good" and sid in self.pipeline.sensors
                    and self.pipeline.sensors[sid].category in categories
                    and sensor_status(self.pipeline.sensors[sid], now) == "online"]

        # site meter = whole-site draw; equipment load = consumers only (a PSU's input power already contains the
        # load it feeds, so adding PSU power to router/radio power would double-count)
        site_power = online_power(("site",))
        load = online_power(("router", "radio_unit"))
        start = now - timedelta(hours=1)
        site_rows = self.store.series("power_consumption", start, now) if site_power else []
        site_rows = [r for r in site_rows if self.pipeline.sensors.get(r[0]) and
                     self.pipeline.sensors[r[0]].category == "site"]
        active = self.alerts.active()
        equipment = self.store.list_equipment()
        monitored = {i.equipment_id for i in infos if i.equipment_id}
        temp = ambient[0] if ambient else (cab[0] if cab else None)
        return {
            "site_id": self.site_id, "site_name": self.settings.site_name,
            "sensors_total": len(infos), "sensors_online": st["online"], "sensors_stale": st["stale"],
            "sensors_offline": st["offline"], "sensors_unknown": st["unknown"], "sensors_disabled": st["disabled"],
            "sensors_simulated": sum(i.data_source == "simulated" for i in infos),
            "sensors_physical": sum(i.data_source == "physical" for i in infos),
            "equipment_total": len(equipment), "equipment_monitored": len(monitored),
            "site_temperature": None if temp is None else {**self.latest_out(temp, now),
                                                          "source": "ambient" if ambient else "cabinet"},
            "site_power_w": round(sum(l.value for l in site_power), 1) if site_power else None,
            "site_power_avg_1h_w": round(sum(r[3] for r in site_rows) / len(site_rows), 1) if site_rows else None,
            "equipment_load_w": round(sum(l.value for l in load), 1) if load else None,
            "equipment_load_sources": len(load),
            "alerts_active": len(active), "alerts_critical": sum(a.severity == "critical" for a in active),
            "alerts_warning": sum(a.severity == "warning" for a in active),
            "alerts_unacknowledged": sum(not a.acknowledged for a in active),
            "readings_per_minute": self.pipeline.readings_per_minute(),
            "ingest": dict(self.pipeline.stats),
            "data_sources": sorted({i.data_source for i in infos}),
            "simulator_running": bool(self.simulator and self.simulator.running),
        }

    def history(self, metric: str, start: datetime, end: datetime, sensor_id: str | None = None,
                equipment_id: str | None = None, max_points: int = 300) -> list[dict]:
        rows = self.store.series(metric, start, end, sensor_id=sensor_id, equipment_id=equipment_id)
        by_sensor: dict[str, list] = defaultdict(list)
        for r in rows:
            by_sensor[r[0]].append(r)
        out = []
        m = METRICS[metric]
        for sid, srows in sorted(by_sensor.items()):
            bucket, points = bucket_series(srows, start, end, max_points)
            out.append({"sensor_id": sid, "equipment_id": srows[-1][1], "metric": metric, "label": m.label,
                        "unit": m.unit, "data_source": srows[-1][4], "bucket_s": bucket, "points": points})
        return out  # empty when there is no data: the dashboard shows no chart

    def availability(self, start: datetime, end: datetime) -> list[dict]:
        """Share of the window each sensor spent online, from recorded status transitions."""
        initial = self.store.last_status_before(start)
        events = self.store.status_events(start)
        per: dict[str, list] = defaultdict(list)
        for e in events:
            per[e.sensor_id].append((e.at, e.status))
        out = []
        span = (end - start).total_seconds()
        for info in self.pipeline.sensors.values():
            sid = info.sensor_id
            cur, t, online = initial.get(sid), start, 0.0
            for at, status in per.get(sid, []):
                if cur == "online":
                    online += (min(at, end) - t).total_seconds()
                cur, t = status, max(at, start)
            if cur == "online":
                online += (end - t).total_seconds()
            known = cur is not None or per.get(sid)
            out.append({"sensor_id": sid, "data_source": info.data_source,
                        "online_share": round(online / span, 4) if known and span > 0 else None,
                        "transitions": len(per.get(sid, []))})
        return out

    def info(self) -> dict:
        rev = None
        try:
            rev = db.current_revision(self.engine) if self.engine is not None else None
        except Exception:  # noqa: BLE001
            pass
        counts = self.store.counts() if self.started else {}
        return {"available": self.started, "error": self.error, "database": self.db_kind, "schema_revision": rev,
                "site_id": self.site_id, "app_env": self.settings.app_env,
                "ingest_api_key_configured": self.settings.telemetry_ingest_api_key is not None,
                "retention_days": self.settings.telemetry_retention_days,
                "measurements_stored": counts.get("measurements"), "oldest_measurement": counts.get("oldest"),
                "websocket_clients": self.hub.clients,
                "mqtt": self.mqtt.status() if self.mqtt else {"enabled": False},
                "simulator": self.simulator.status() if self.simulator else None,
                "recent_ingest_errors": list(self.pipeline.recent_errors)[-10:] if self.pipeline else [],
                "communication_rule": COMMUNICATION_RULE}
