"""Real-time anomaly engine: telemetry readings -> features -> detectors -> context -> anomaly episodes.

Evaluation is driven by MEASUREMENT time: a series is evaluated when one of its 1-minute buckets closes. Live data,
simulator backfill and offline replays therefore behave identically, and expensive work happens at most once per
bucket per series (not per reading).

Episodes (one record per continuous condition):
  - key = (equipment, sensor, metric, univariate) or (equipment, -, -, multivariate); detectors that fire on the same
    key are grouped into the same episode (`detectors` lists them all)
  - opens after `min_consecutive` evaluations >= 1.0; severity = critical at >= severity.critical
  - ends after `clear_after` evaluations below `clear_level`; auto-resolved by "system" if configured
  - a recurrence within `cooldown_s` re-opens the same record (occurrences += 1) instead of creating a duplicate
Gaps (sensor silent / offline) are never evaluated and never end or open an episode: communication problems are not
equipment anomalies. Context: metrics listed in `mains_outage_expected` are not flagged while mains power is lost,
while the battery recharges afterwards (`recharge_expected`; charging detected from battery_state/current).
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from app.analytics import detectors as det
from app.analytics.config import detector_params, metric_config
from app.analytics.features import Features, SeriesState
from app.analytics.models import Anomaly
from app.telemetry.catalog import METRICS

logger = logging.getLogger(__name__)
SITE = "site"


def mv_vector(by_metric: dict[str, Features], metrics: list[str]) -> tuple[list[str], np.ndarray] | None:
    """Multivariate feature vector: the bucket mean of each metric (None if any is missing). Shared by training and
    inference. Values only: adding 10-minute changes diluted the isolation in tests (unusual radio-unit
    temperature/power pairs scored 0.3-0.4 instead of 1.3-1.6 threshold units); sudden changes are already covered
    by the univariate rate-of-change detector."""
    names, vals = [], []
    for m in metrics:
        f = by_metric.get(m)
        if f is None or f.value is None:
            return None
        names.append(f"{m}:value")
        vals.append(f.value)
    return names, np.asarray(vals, float)


@dataclass
class Meta:
    equipment_id: str | None
    sensor_id: str
    metric: str
    category: str
    data_source: str
    detectors: list[str]
    unit: str
    label: str
    state: bool


@dataclass
class Episode:
    anomaly: Anomaly
    below: int = 0
    updates: int = 0


@dataclass
class LoadedModel:
    model_id: int
    model: object
    features: list[str]
    median: np.ndarray
    scale: np.ndarray
    threshold: float
    center: float


@dataclass
class MvState:
    bucket: int | None = None
    by_metric: dict = field(default_factory=dict)


class AnalyticsEngine:
    def __init__(self, cfg: dict, store=None, site_id: str = "SITE-001", publish=lambda kind, payload: None,
                 mv_sink=None):
        self.cfg, self.store, self.site_id, self.publish = cfg, store, site_id, publish
        self.mv_sink = mv_sink  # training: receives (category, data_source, equipment, t, names, vector)
        self.quiet = False  # warm start: rebuild feature windows from history without creating anomalies
        self._lock = threading.RLock()
        self.series: dict[tuple, SeriesState] = {}
        self.meta: dict[tuple, Meta] = {}
        self.episodes: dict[tuple, Episode] = {}
        self.pending: dict[tuple, int] = defaultdict(int)
        self.mv: dict[str, MvState] = defaultdict(MvState)
        self.models: dict[tuple[str, str], LoadedModel] = {}
        self.mains: bool | None = None  # latest site mains state (None = no mains sensor)
        self.mains_restored_at: datetime | None = None
        self.mains_lost_at: datetime | None = None
        self.charging = False  # battery recharging (from battery_state / battery_current)
        self.charge_ended_at: datetime | None = None
        self.stats = {"readings": 0, "evaluations": 0, "eval_seconds": 0.0, "suppressed": 0, "opened": 0,
                      "reopened": 0, "ended": 0, "mv_scored": 0}
        if store is not None:
            for a in store.unresolved_anomalies():
                if a.condition_active:
                    self.episodes[self._key(a.equipment_id, a.sensor_id, a.metric, a.kind)] = Episode(a)

    # --- configuration / models ---------------------------------------------------------------------
    def reconfigure(self, cfg: dict) -> None:
        """New configuration: feature windows restart from scratch (warm-up applies again)."""
        with self._lock:
            self.cfg = cfg
            self.series.clear()
            self.meta.clear()
            self.pending.clear()
            self.mv.clear()

    def load_models(self, rows, base_dir: Path) -> list[str]:
        import joblib

        loaded = []
        with self._lock:
            self.models.clear()
            for r in rows:
                path = Path(r.artifact_path)
                path = path if path.is_absolute() else base_dir / path
                try:
                    a = joblib.load(path)
                except Exception as exc:  # noqa: BLE001 - a broken artifact disables only that model
                    logger.error("Anomaly model %s (%s) could not be loaded: %s", r.id, path.name, exc)
                    continue
                self.models[(r.category, r.data_source)] = LoadedModel(
                    r.id, a["model"], a["features"], np.asarray(a["median"]), np.asarray(a["scale"]),
                    float(a["threshold"]), float(a["center"]))
                loaded.append(f"{r.category}/{r.data_source}#{r.id}")
        return loaded

    # --- ingestion hook ----------------------------------------------------------------------------
    @staticmethod
    def _key(equipment_id, sensor_id, metric, kind="univariate") -> tuple:
        return (equipment_id or SITE, sensor_id if kind == "univariate" else None,
                metric if kind == "univariate" else None, kind)

    def on_reading(self, sensor_id: str, equipment_id: str | None, category: str, data_source: str, metric: str,
                   value: float | None, ts: datetime, quality: str = "good") -> None:
        if quality != "good" or value is None:
            return
        with self._lock:
            self.stats["readings"] += 1
            if metric == "mains_available":
                self._mains(bool(value >= 0.5), ts)
            elif metric == "battery_state":
                self._charging(value >= 0.5, ts)
            elif metric == "battery_current":
                self._charging(value > self.cfg.get("context", {}).get("charge_current_a", 2.0), ts)
            dets = metric_config(self.cfg, category, metric)
            if not dets:
                return
            key = self._key(equipment_id, sensor_id, metric)
            st = self.series.get(key)
            if st is None:
                fp = detector_params(self.cfg, "forecast_residual")
                st = SeriesState(self.cfg["grid_s"], self.cfg["baseline_buckets"], self.cfg["short_buckets"],
                                 self.cfg["max_gap_buckets"], self.cfg.get("anomalous_share_relearn", 0.5),
                                 fp.get("alpha", 0.3), fp.get("beta", 0.05))
                m = METRICS[metric]
                self.series[key] = st
                self.meta[key] = Meta(equipment_id, sensor_id, metric, category, data_source, list(dets), m.unit,
                                      m.label, m.kind == "state")
            if st.add(ts, float(value)):
                t0 = time.perf_counter()
                self._evaluate(key, st, self.meta[key])
                self.stats["eval_seconds"] += time.perf_counter() - t0

    def _mains(self, available: bool, ts: datetime) -> None:
        if self.mains is not None and available != self.mains:
            if available:
                self.mains_restored_at = ts
            else:
                self.mains_lost_at = ts
        self.mains = available

    def _charging(self, charging: bool, ts: datetime) -> None:
        if self.charging and not charging:
            self.charge_ended_at = ts
        self.charging = charging

    def suppressed_by_context(self, category: str, metric: str, t: datetime) -> str | None:
        spec = self.cfg["categories"].get(category) or {}
        outage, recharge = spec.get("mains_outage_expected") or [], spec.get("recharge_expected") or []
        if (metric not in outage and metric not in recharge) or self.mains is None:
            return None
        if self.mains is False:
            return "mains power lost: backup-power behaviour is expected"
        if metric not in recharge or self.mains_restored_at is None:
            return None
        ctx = self.cfg.get("context", {})
        since = (t - self.mains_restored_at).total_seconds()
        if since < 0 or since > ctx.get("after_mains_restored_s", 21600):
            return None
        settled = (self.charge_ended_at is not None and self.charge_ended_at >= self.mains_restored_at
                   and (t - self.charge_ended_at).total_seconds() > ctx.get("recharge_settle_s", 3600))
        if since <= ctx.get("min_after_restore_s", 1800) or self.charging or not settled:
            return "battery recharging after a mains outage: charge current, PSU load and battery heating are expected"
        return None

    # --- evaluation ------------------------------------------------------------------------------------
    def warmed(self, key: tuple) -> bool:
        st = self.series.get(key)
        f = st.features(self.meta[key].state) if st else None
        return bool(f and f.n_history >= self.cfg["min_history_buckets"])

    def _evaluate(self, key: tuple, st: SeriesState, meta: Meta) -> None:
        f = st.features(state_metric=meta.state)
        if f is None or f.value is None or f.n_history < self.cfg["min_history_buckets"]:
            return  # gap or warm-up: no judgement
        self.stats["evaluations"] += 1
        floor = self.cfg["scale_floor"].get(meta.metric, 1e-6)
        findings = []
        for name in meta.detectors:
            fn = det.UNIVARIATE[name]
            r = fn(f, detector_params(self.cfg, name, meta.metric), floor, meta.unit, meta.label)
            if r is not None:
                findings.append(r)
        if not findings:
            return
        context = self.suppressed_by_context(meta.category, meta.metric, f.t)
        if context:
            self.stats["suppressed"] += 1
            st.reset_forecast()  # restart the forecast cleanly once the expected behaviour is over
        top = max(findings, key=lambda x: x.score)
        if top.score >= self.cfg["severity"]["warning"]:
            st.flag_last()  # keep abnormal (or context-explained) buckets out of the baseline
        self._episode(key, meta, f, top, findings, context, None)
        if meta.equipment_id:
            self._multivariate(meta, st, f)

    def _multivariate(self, meta: Meta, st: SeriesState, f: Features) -> None:
        spec = (self.cfg["categories"].get(meta.category) or {}).get("multivariate") or []
        if meta.metric not in spec:
            return
        mv = self.mv[meta.equipment_id]
        if mv.bucket != st.last_index:
            mv.bucket, mv.by_metric = st.last_index, {}
        mv.by_metric[meta.metric] = f
        if set(spec) - set(mv.by_metric):
            return
        vec = mv_vector(mv.by_metric, spec)
        mv.by_metric = {}
        if vec is None:
            return
        names, x = vec
        if any(self.suppressed_by_context(meta.category, m, f.t) for m in spec):
            return  # expected behaviour (outage / recharge): neither scored nor used for training
        if self.mv_sink is not None:
            self.mv_sink(meta.category, meta.data_source, meta.equipment_id, f.t, names, x)
            return
        model = self.models.get((meta.category, meta.data_source))  # never across physical/simulated
        if model is None or model.features != names:
            return
        a = -float(model.model.score_samples(x.reshape(1, -1))[0])
        score = max(0.0, (a - model.center) / max(model.threshold - model.center, 1e-9))
        self.stats["mv_scored"] += 1
        z = (x - model.median) / model.scale
        top = np.argsort(-np.abs(z))[:3]
        parts = [f"{names[i]} = {x[i]:.3g} ({z[i]:+.1f} robust SD from training median {model.median[i]:.3g})"
                 for i in top]
        finding = det.Finding("isolation_forest", min(score, det.MAX_SCORE), None, None, None,
                              "Unusual combination of measurements; largest deviations: " + "; ".join(parts))
        key = self._key(meta.equipment_id, None, None, "multivariate")
        mmeta = Meta(meta.equipment_id, None, None, meta.category, meta.data_source, ["isolation_forest"], "", "", False)
        evidence_f = Features(t=f.t, value=None, n_history=f.n_history, availability=f.availability)
        self._episode(key, mmeta, evidence_f, finding, [finding], None, model.model_id,
                      extra={"features": dict(zip(names, map(float, x))),
                             "robust_z": dict(zip(names, map(float, z)))})

    def _severity(self, score: float) -> str:
        return "critical" if score >= self.cfg["severity"]["critical"] else "warning"

    def _episode(self, key, meta: Meta, f: Features, top, findings, context, model_id, extra: dict | None = None):
        if self.quiet:
            return
        ep_cfg = self.cfg["episode"]
        warn = self.cfg["severity"]["warning"]
        score = 0.0 if context else top.score
        ep = self.episodes.get(key)
        t = f.t
        evidence = {"features": f.as_dict(), "findings": [{"detector": x.detector, "score": round(x.score, 3),
                                                            "detail": x.detail} for x in findings]}
        if context:
            evidence["context"] = context
        if extra:
            evidence.update(extra)

        if ep is not None and ep.anomaly.condition_active:
            a = ep.anomaly
            if score >= ep_cfg["clear_level"]:
                ep.below = 0
                a.last_detected_at, a.score = t, round(score, 3)
                if score >= warn:
                    a.detectors = sorted(set(a.detectors) | {x.detector for x in findings if x.score >= warn})
                changed = False
                if score > a.peak_score:
                    a.peak_score = round(score, 3)
                    a.observed, a.expected, a.explanation, a.detector = top.observed, top.expected, top.detail, top.detector
                    a.evidence = evidence
                if self._severity(score) == "critical" and a.severity != "critical" and score >= warn:
                    a.severity, changed = "critical", True
                ep.updates += 1
                if changed or ep.updates >= 10:
                    ep.updates = 0
                    self._save(ep, "anomaly_updated" if not changed else "anomaly_escalated")
            else:
                ep.below += 1
                if ep.below >= ep_cfg["clear_after"]:
                    a.condition_active, a.ended_at, a.score = False, t, round(score, 3)
                    if ep_cfg.get("auto_resolve", True) and a.status != "resolved":
                        a.status, a.resolved_at, a.resolved_by = "resolved", t, "system"
                        a.resolution_note = a.resolution_note or "condition returned to normal"
                    self.stats["ended"] += 1
                    self._save(ep, "anomaly_ended")
            return

        if score >= warn:
            self.pending[key] += 1
        else:
            self.pending[key] = 0
            return
        if self.pending[key] < ep_cfg["min_consecutive"]:
            return
        self.pending[key] = 0
        if ep is not None and ep.anomaly.ended_at and (t - ep.anomaly.ended_at).total_seconds() <= ep_cfg["cooldown_s"]:
            a = ep.anomaly  # recurrence: re-open the same record
            a.condition_active, a.ended_at, a.occurrences = True, None, a.occurrences + 1
            a.last_detected_at, a.score = t, round(score, 3)
            a.peak_score = max(a.peak_score, round(score, 3))
            a.severity = "critical" if "critical" in (a.severity, self._severity(score)) else "warning"
            if a.status == "resolved":
                a.status, a.resolved_at, a.resolved_by = "open", None, None
            ep.below = 0
            self.stats["reopened"] += 1
            self._save(ep, "anomaly_reopened")
            return
        a = Anomaly(site_id=self.site_id, equipment_id=meta.equipment_id, sensor_id=meta.sensor_id,
                    category=meta.category, metric=meta.metric, kind=key[3], detector=top.detector,
                    detectors=sorted({x.detector for x in findings if x.score >= warn}), model_id=model_id,
                    observed=top.observed, expected=top.expected, unit=meta.unit or None, score=round(score, 3),
                    peak_score=round(score, 3), severity=self._severity(score), status="open", condition_active=True,
                    first_detected_at=t, last_detected_at=t, occurrences=1, explanation=top.detail,
                    evidence=evidence, data_source=meta.data_source)
        self.episodes[key] = Episode(a)
        self.stats["opened"] += 1
        self._save(self.episodes[key], "anomaly_opened")

    def _save(self, ep: Episode, event: str) -> None:
        if self.store is not None:
            ep.anomaly = self.store.save_anomaly(ep.anomaly)
        self.publish("anomaly", {"event": event, "anomaly": ep.anomaly})

    def operator_resolved(self, anomaly: Anomaly) -> None:
        """An operator resolved an anomaly: forget its episode if the condition has ended."""
        with self._lock:
            for k, ep in list(self.episodes.items()):
                if ep.anomaly.id == anomaly.id:
                    ep.anomaly.status, ep.anomaly.resolved_by = anomaly.status, anomaly.resolved_by
                    ep.anomaly.resolved_at = anomaly.resolved_at

    # --- read helpers for health --------------------------------------------------------------------
    def coverage(self, equipment_id: str, category: str, sensor_metrics: dict[str, list[str]],
                 online: set[str]) -> dict:
        """Which of the category's configured metrics this equipment can be judged on right now."""
        configured = list(((self.cfg["categories"].get(category) or {}).get("metrics") or {}).keys())
        provided = {m for metrics in sensor_metrics.values() for m in metrics}
        usable, detail = [], {}
        with self._lock:
            for m in configured:
                if m not in provided:
                    detail[m] = "no sensor"
                    continue
                sids = [s for s, ms in sensor_metrics.items() if m in ms]
                fresh = [s for s in sids if s in online]
                warm = [s for s in fresh if self.warmed(self._key(equipment_id, s, m))]
                detail[m] = "ok" if warm else ("warming up" if fresh else "sensor not online")
                if warm:
                    usable.append(m)
        expected = [m for m in configured if m in provided]
        return {"configured": configured, "provided": expected, "usable": usable, "detail": detail,
                "coverage": round(len(usable) / len(expected), 3) if expected else 0.0}

    def benchmark(self) -> dict:
        n = self.stats["evaluations"]
        return {**self.stats, "eval_ms_mean": round(1000 * self.stats["eval_seconds"] / n, 4) if n else None,
                "series": len(self.series)}


def since(hours: float, now: datetime) -> datetime:
    return now - timedelta(hours=hours)
