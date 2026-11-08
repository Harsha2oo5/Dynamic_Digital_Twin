"""AnalyticsService: wires the anomaly engine into telemetry ingestion, keeps equipment health up to date, stores
health snapshots and runs the maintenance analysis. Attached to app.state.analytics.

It depends on the telemetry service (same database). If telemetry is unavailable, analytics is unavailable too and
its endpoints answer 503; detection and telemetry are unaffected by analytics failures.
"""

import logging
import threading
from datetime import UTC, datetime, timedelta

from app.analytics import config as acfg
from app.analytics import health as health_mod
from app.analytics import maintenance
from app.analytics.engine import AnalyticsEngine
from app.analytics.store import AnalyticsStore
from app.analytics.train import replay
from app.core.config import BACKEND_DIR
from app.telemetry.ingest import Latest
from app.telemetry.service import TelemetryService, sensor_status

logger = logging.getLogger(__name__)


class AnalyticsUnavailable(RuntimeError):
    pass


class AnalyticsService:
    def __init__(self, telemetry: TelemetryService, worker_interval_s: float = 30.0):
        self.t = telemetry
        self.worker_interval_s = worker_interval_s
        self.started = False
        self.error: str | None = None
        self.store: AnalyticsStore | None = None
        self.engine: AnalyticsEngine | None = None
        self.cfg: dict | None = None
        self.config_version: int | None = None
        self.health: dict[str, dict] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_snapshot: datetime | None = None
        self._last_maintenance: datetime | None = None
        self._lock = threading.Lock()
        self.loaded_models: list[str] = []

    # --- lifecycle -------------------------------------------------------------------------------------
    def start(self, warm_start: bool = True, worker: bool = True) -> None:
        if not self.t.started:
            self.error = f"telemetry unavailable: {self.t.error}"
            return
        try:
            self.store = AnalyticsStore(self.t.store.sessions)
            active = self.store.active_config()
            if active is None:
                cfg, text = acfg.load()
                active = self.store.add_config(text, acfg.config_hash(cfg), "default file", "seeded from "
                                               "configs/analytics/anomaly.yaml")
            self.cfg, self.config_version = acfg.parse(active.content), active.version
            self.engine = AnalyticsEngine(self.cfg, self.store, self.t.site_id, self._publish)
            self.loaded_models = self.engine.load_models(self.store.models(active_only=True), BACKEND_DIR)
            if warm_start:
                self.warm_start()
        except Exception as exc:  # noqa: BLE001
            self.error = f"analytics failed to start: {type(exc).__name__}: {exc}"
            logger.exception("Analytics start failed")
            return
        self.t.listeners.append(self.on_measurements)
        self.started = True
        if worker:
            self._thread = threading.Thread(target=self._work, name="analytics-worker", daemon=True)
            self._thread.start()
        logger.info("Analytics started (config v%s, models: %s)", self.config_version, self.loaded_models or "none")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(10)
        if self.on_measurements in self.t.listeners:
            self.t.listeners.remove(self.on_measurements)

    def require(self) -> None:
        if not self.started:
            raise AnalyticsUnavailable(self.error or "analytics is starting")

    def warm_start(self, now: datetime | None = None) -> int:
        """Rebuild feature windows from the last baseline window of stored telemetry (no anomalies are created)."""
        now = now or datetime.now(UTC)
        span = timedelta(seconds=self.cfg["grid_s"] * (self.cfg["baseline_buckets"] + self.cfg["short_buckets"] + 5))
        self.engine.quiet = True
        try:
            n = sum(replay(self.t.store.sessions, self.engine, now - span, now, src) for src in ("physical", "simulated"))
        finally:
            self.engine.quiet = False
        return n

    # --- ingestion hook ----------------------------------------------------------------------------
    def on_measurements(self, readings: list[Latest]) -> None:
        sensors = self.t.pipeline.sensors
        for r in readings:
            info = sensors.get(r.sensor_id)
            if info is not None:
                self.engine.on_reading(r.sensor_id, info.equipment_id, info.category, info.data_source, r.metric,
                                       r.value, r.ts, r.quality)

    def _publish(self, kind: str, payload: dict) -> None:
        if kind == "anomaly":
            self.t.hub.publish({"type": "anomaly", "event": payload["event"], "anomaly": self.anomaly_out(payload["anomaly"])})
        else:
            self.t.hub.publish({"type": kind, **payload})

    # --- worker ------------------------------------------------------------------------------------------
    def _work(self) -> None:
        while not self._stop.wait(self.worker_interval_s):
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                logger.exception("Analytics worker iteration failed")

    def tick(self, now: datetime | None = None, force: bool = False) -> None:
        now = now or datetime.now(UTC)
        m = self.cfg["maintenance"]
        if force or self._last_maintenance is None or (now - self._last_maintenance).total_seconds() >= m["run_every_s"]:
            self._last_maintenance = now
            self.run_maintenance(now)
        self.refresh_health(now)
        h = self.cfg["health"]
        if force or self._last_snapshot is None or (now - self._last_snapshot).total_seconds() >= h["snapshot_every_s"]:
            self._last_snapshot = now
            self.store.add_health([{"equipment_id": r["equipment_id"], "at": now, "score": r["score"],
                                    "status": r["status"], "coverage": r["coverage"], "data_quality": r["data_quality"],
                                    "components": {"components": r["components"][:20],
                                                   "unknown_reason": r["unknown_reason"]}}
                                   for r in self.health.values()])

    # --- health -----------------------------------------------------------------------------------------
    def _equipment_sensors(self, equipment_id: str, now: datetime):
        infos = [i for i in self.t.pipeline.sensors.values() if i.equipment_id == equipment_id]
        return ({i.sensor_id: sorted(i.metrics) for i in infos},
                {i.sensor_id for i in infos if sensor_status(i, now) == "online"}, infos)

    def compute_health(self, eq, now: datetime, availability: dict[str, float | None] | None = None) -> dict:
        sensor_metrics, online, infos = self._equipment_sensors(eq.equipment_id, now)
        cov = self.engine.coverage(eq.equipment_id, eq.category, sensor_metrics, online)
        anomalies = self.store.recent_anomalies(now - timedelta(hours=self.cfg["health"]["lookback_h"]),
                                                eq.equipment_id)
        alerts = [a for a in self.t.alerts.active() if a.equipment_id == eq.equipment_id]
        recs = self.store.active_recommendations(eq.equipment_id)
        if infos:
            shares = [(availability or {}).get(i.sensor_id) for i in infos]
            shares = [s for s in shares if s is not None]
            dq = sum(shares) / len(shares) if shares else len(online) / len(infos)
        else:
            dq = None
        r = health_mod.compute(eq.equipment_id, eq.category, cov, anomalies, alerts, recs, dq, self.cfg, now)
        r.update(name=eq.name, data_source=eq.data_source, sensors=len(infos), sensors_online=len(online))
        return r

    def refresh_health(self, now: datetime | None = None) -> dict[str, dict]:
        now = now or datetime.now(UTC)
        avail = {a["sensor_id"]: a["online_share"] for a in self.t.availability(now - timedelta(hours=24), now)}
        new = {eq.equipment_id: self.compute_health(eq, now, avail) for eq in self.t.store.list_equipment()}
        with self._lock:
            old, self.health = self.health, new
        for eid, r in new.items():
            prev = old.get(eid)
            if prev is None or prev["status"] != r["status"] or prev["score"] != r["score"]:
                self.t.hub.publish({"type": "health", "equipment_id": eid, "status": r["status"], "score": r["score"],
                                    "coverage": r["coverage"], "data_quality": r["data_quality"]})
        return new

    def health_of(self, equipment_id: str) -> dict | None:
        with self._lock:
            r = self.health.get(equipment_id)
        if r is None:
            eq = self.t.store.get_equipment(equipment_id)
            if eq is None:
                return None
            r = self.compute_health(eq, datetime.now(UTC))
        return r

    def summary(self) -> dict:
        with self._lock:
            rows = list(self.health.values())
        counts = {s: sum(r["status"] == s for r in rows) for s in ("healthy", "warning", "degraded", "critical",
                                                                    "unknown")}
        scores = [r["score"] for r in rows if r["score"] is not None]
        active_total, _ = self.store.anomalies_page(1, 0, active=True)
        open_total, _ = self.store.anomalies_page(1, 0, status="open")
        inspection = [r for r in rows if r["status"] in ("degraded", "critical")]
        recs = self.store.active_recommendations()
        return {"equipment": len(rows), **counts,
                "mean_score": round(sum(scores) / len(scores), 1) if scores else None,
                "active_anomalies": active_total, "open_anomalies": open_total,
                "active_recommendations": len(recs),
                "requiring_inspection": sorted({r["equipment_id"] for r in inspection} | {x.equipment_id for x in recs
                                                                                         if x.evidence_level != "low"}),
                "config_version": self.config_version, "models": self.loaded_models}

    # --- maintenance ------------------------------------------------------------------------------------
    def run_maintenance(self, now: datetime | None = None) -> list:
        now = now or datetime.now(UTC)
        window = timedelta(hours=self.cfg["maintenance"]["window_h"])
        mains = self.t.store.series("mains_available", now - window, now)
        changed = []
        for eq in self.t.store.list_equipment():
            sensor_metrics, _, _ = self._equipment_sensors(eq.equipment_id, now)
            provided = {m for ms in sensor_metrics.values() for m in ms}
            if not provided:
                continue
            times = self.store.anomaly_counts_by_day(eq.equipment_id, now - timedelta(days=7))
            related = self.store.recent_anomalies(now - window, eq.equipment_id)
            mrows = [r for r in mains if r[4] == eq.data_source]
            cands, assessed = maintenance.analyse_equipment(eq, provided, self.t.store.series, self.cfg, now, mrows,
                                                           times, related)
            changed += maintenance.apply(self.store, self.t.site_id, eq, cands, assessed, now)
        for r in changed:
            self.t.hub.publish({"type": "recommendation", "recommendation": self.rec_out(r)})
        return changed

    # --- configuration / models -------------------------------------------------------------------------
    def update_config(self, text: str, by: str | None, note: str | None) -> int:
        cfg = acfg.parse(text)  # raises ConfigError
        row = self.store.add_config(text, acfg.config_hash(cfg), by, note)
        self.cfg, self.config_version = cfg, row.version
        self.engine.reconfigure(cfg)
        self.warm_start()
        return row.version

    def reload_models(self) -> list[str]:
        self.loaded_models = self.engine.load_models(self.store.models(active_only=True), BACKEND_DIR)
        return self.loaded_models

    # --- serializers ------------------------------------------------------------------------------------
    @staticmethod
    def _iso(v):
        return v.isoformat() if isinstance(v, datetime) else v

    def anomaly_out(self, a) -> dict:
        cols = ("id", "equipment_id", "sensor_id", "category", "metric", "kind", "detector", "detectors", "model_id",
                "observed", "expected", "unit", "score", "peak_score", "severity", "status", "condition_active",
                "first_detected_at", "last_detected_at", "ended_at", "occurrences", "explanation", "evidence",
                "data_source", "acknowledged_at", "acknowledged_by", "resolved_at", "resolved_by", "resolution_note")
        return {c: self._iso(getattr(a, c)) for c in cols}

    def rec_out(self, r) -> dict:
        cols = ("id", "equipment_id", "category", "condition_key", "condition", "metric", "action", "evidence_level",
                "evidence", "status", "acknowledged", "acknowledged_by", "acknowledged_at", "data_source",
                "created_at", "updated_at", "resolved_at")
        return {c: self._iso(getattr(r, c)) for c in cols}

    def health_out(self, r: dict) -> dict:
        return {**r, "at": self._iso(r["at"])}
