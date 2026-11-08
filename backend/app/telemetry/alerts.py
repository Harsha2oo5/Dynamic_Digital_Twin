"""Threshold-based alerting (no machine learning in this milestone).

Alerts are keyed by (rule, sensor): one active alert per rule and sensor. An alert opens when a good, in-order
reading crosses a level for at least `for_s`, escalates or de-escalates between warning and critical while active,
and clears (with hysteresis) when the value returns to normal. Acknowledgement is independent of clearing.
Messages describe a detected condition that needs investigation; they never declare a component faulty.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from app.telemetry.catalog import METRICS
from app.telemetry.models import Alert

COMMUNICATION_RULE = "sensor_communication"


@dataclass(frozen=True)
class Rule:
    id: str
    metric: str
    direction: str  # above | below
    warning: float | None = None
    critical: float | None = None
    hysteresis: float = 0.0
    for_s: float = 0.0
    categories: tuple[str, ...] = ()

    def level(self, value: float) -> tuple[str | None, float | None]:
        """(severity, threshold crossed) for `value`; (None, None) when normal."""
        crossed = (lambda t: value > t) if self.direction == "above" else (lambda t: value < t)
        if self.critical is not None and crossed(self.critical):
            return "critical", self.critical
        if self.warning is not None and crossed(self.warning):
            return "warning", self.warning
        return None, None

    def cleared(self, value: float) -> bool:
        """True when `value` is back to normal including the hysteresis margin."""
        base = self.warning if self.warning is not None else self.critical
        if self.direction == "above":
            return value <= base - self.hysteresis
        return value >= base + self.hysteresis


@dataclass
class AlertConfig:
    rules: list[Rule] = field(default_factory=list)
    communication_enabled: bool = True
    communication_critical_after_s: float = 900.0

    def rules_for(self, metric: str, category: str | None) -> list[Rule]:
        return [r for r in self.rules if r.metric == metric and (not r.categories or category in r.categories)]


def load_rules(path: Path | None) -> AlertConfig:
    """Parse and validate the YAML rules. Errors name the offending rule."""
    if path is None or not Path(path).is_file():
        return AlertConfig()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    rules, seen = [], set()
    for raw in data.get("rules", []):
        rid = raw.get("id")
        if not rid or rid in seen:
            raise ValueError(f"alert rule without id or duplicate id: {raw}")
        seen.add(rid)
        if raw.get("metric") not in METRICS:
            raise ValueError(f"rule {rid}: unknown metric {raw.get('metric')!r}")
        dirs = [d for d in ("above", "below") if d in raw]
        if len(dirs) != 1:
            raise ValueError(f"rule {rid}: give exactly one of 'above' or 'below'")
        levels = raw[dirs[0]] or {}
        if not set(levels) <= {"warning", "critical"} or not levels:
            raise ValueError(f"rule {rid}: levels must be warning and/or critical")
        w, c = levels.get("warning"), levels.get("critical")
        if w is not None and c is not None and ((dirs[0] == "above" and c < w) or (dirs[0] == "below" and c > w)):
            raise ValueError(f"rule {rid}: critical level must be beyond the warning level")
        rules.append(Rule(id=rid, metric=raw["metric"], direction=dirs[0], warning=w, critical=c,
                          hysteresis=float(raw.get("hysteresis", 0.0)), for_s=float(raw.get("for_s", 0.0)),
                          categories=tuple(raw.get("categories", ()))))
    comm = data.get("communication", {}) or {}
    return AlertConfig(rules, bool(comm.get("enabled", True)), float(comm.get("critical_after_s", 900)))


def _fmt(v: float | None, unit: str) -> str:
    if v is None:
        return "n/a"
    return f"{v:g} {unit}".strip() if unit not in ("state", "ratio") else f"{v:g}"


class AlertEngine:
    """Keeps active alerts in memory (mirrored to the database) and evaluates readings and sensor status."""

    def __init__(self, config: AlertConfig, store, site_id: str, publish=lambda kind, payload: None):
        self.config, self.store, self.site_id, self.publish = config, store, site_id, publish
        self._lock = threading.Lock()
        self._active: dict[tuple[str, str], Alert] = {}
        self._pending: dict[tuple[str, str], datetime] = {}  # first violation time, for for_s
        if store is not None:
            for a in store.active_alerts():
                self._active[(a.rule_id, a.sensor_id or "")] = a

    def active(self) -> list[Alert]:
        with self._lock:
            return list(self._active.values())

    def _save(self, alert: Alert) -> Alert:
        saved = self.store.save_alert(alert) if self.store is not None else alert
        self.publish("alert", saved)
        return saved

    def evaluate(self, sensor_id: str, equipment_id: str | None, category: str | None, metric: str, value: float,
                 ts: datetime, data_source: str) -> list[Alert]:
        """Evaluate one good, in-order reading. Returns alerts that were opened, changed or cleared."""
        changed = []
        unit = METRICS[metric].unit
        with self._lock:
            for rule in self.config.rules_for(metric, category):
                key = (rule.id, sensor_id)
                severity, threshold = rule.level(value)
                alert = self._active.get(key)
                if alert is not None:
                    alert.last_triggered_at = ts if severity else alert.last_triggered_at
                    worse = (value > (alert.peak_value or value)) if rule.direction == "above" else (
                        value < (alert.peak_value or value))
                    if worse:
                        alert.peak_value = value
                    if severity and severity != alert.severity:
                        alert.severity, alert.threshold, alert.value = severity, threshold, value
                        alert.message = self._message(rule, value, threshold, unit, severity, data_source)
                        self._active[key] = self._save(alert)
                        changed.append(self._active[key])
                    elif severity is None and rule.cleared(value):
                        alert.state, alert.cleared_at, alert.value = "cleared", ts, value
                        changed.append(self._save(alert))
                        del self._active[key]
                        self._pending.pop(key, None)
                    else:
                        alert.value = value
                    continue
                if severity is None:
                    self._pending.pop(key, None)
                    continue
                first = self._pending.setdefault(key, ts)
                if (ts - first).total_seconds() < rule.for_s:
                    continue
                alert = Alert(rule_id=rule.id, severity=severity, state="active", site_id=self.site_id,
                              sensor_id=sensor_id, equipment_id=equipment_id, metric=metric, value=value,
                              peak_value=value, threshold=threshold, data_source=data_source, opened_at=ts,
                              last_triggered_at=ts, acknowledged=False,
                              message=self._message(rule, value, threshold, unit, severity, data_source))
                self._active[key] = self._save(alert)
                changed.append(self._active[key])
        return changed

    @staticmethod
    def _message(rule: Rule, value: float, threshold: float, unit: str, severity: str, data_source: str) -> str:
        label = METRICS[rule.metric].label
        states = METRICS[rule.metric].states
        shown = states.get(int(value), _fmt(value, unit)) if states else _fmt(value, unit)
        rel = "above" if rule.direction == "above" else "below"
        text = (f"Detected condition: {label.lower()} {shown} is {rel} the {severity} level "
                f"{_fmt(threshold, unit) if not states else ''}".rstrip() + ". Requires investigation.")
        return text + (" [simulated data]" if data_source == "simulated" else "")

    def sensor_status(self, sensor_id: str, equipment_id: str | None, status: str, offline_since: datetime | None,
                      now: datetime, data_source: str) -> list[Alert]:
        """Communication alerts from the sensor's connection state."""
        if not self.config.communication_enabled:
            return []
        key = (COMMUNICATION_RULE, sensor_id)
        changed = []
        with self._lock:
            alert = self._active.get(key)
            if status == "offline":
                age = (now - offline_since).total_seconds() if offline_since else 0.0
                severity = "critical" if age >= self.config.communication_critical_after_s else "warning"
                msg = (f"Detected condition: sensor {sensor_id} has stopped reporting (offline). Its last readings are "
                       f"shown as last known values. Requires investigation."
                       + (" [simulated data]" if data_source == "simulated" else ""))
                if alert is None:
                    alert = Alert(rule_id=COMMUNICATION_RULE, severity=severity, state="active", site_id=self.site_id,
                                  sensor_id=sensor_id, equipment_id=equipment_id, metric=None, value=None,
                                  peak_value=None, threshold=None, message=msg, data_source=data_source,
                                  opened_at=now, last_triggered_at=now, acknowledged=False)
                    self._active[key] = self._save(alert)
                    changed.append(self._active[key])
                elif alert.severity != severity:
                    alert.severity, alert.last_triggered_at = severity, now
                    self._active[key] = self._save(alert)
                    changed.append(self._active[key])
            elif status == "online" and alert is not None:
                alert.state, alert.cleared_at = "cleared", now
                changed.append(self._save(alert))
                del self._active[key]
        return changed

    def forget_sensor(self, sensor_id: str) -> None:
        with self._lock:
            for key in [k for k in self._active if k[1] == sensor_id]:
                del self._active[key]
            for key in [k for k in self._pending if k[1] == sensor_id]:
                del self._pending[key]
