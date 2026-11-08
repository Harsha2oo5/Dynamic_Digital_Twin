"""Simulation-based validation of anomaly detection, health scoring and maintenance recommendations.

SIMULATED DATA ONLY. The detectors are evaluated against faults that the simulator itself injects; the numbers show
that the pipeline separates these simulated patterns from simulated normal operation. They are NOT real-world
accuracy, and they say nothing about real equipment failures.

Timeline (simulated time, 1-minute steps, random mains outages and door openings stay ON):
  day 0-3   normal operation; Isolation Forest models are trained on it afterwards
  day 3-4   normal operation (false-alarm measurement with all detectors, including the trained models)
  day 4     sudden faults with recovery: radio unit temperature spike, radio unit power anomaly, PSU voltage
            fluctuation, and a sensor communication failure (radio unit sensor silent)
  day 5-7   gradual faults: battery degradation (48 h) and PSU overheating (36 h)

From backend/:
    python -m app.analytics.validate --name v1
Writes experiments/anomaly_validation/<name>/report.{json,md}.
"""

import argparse
import json
import logging
import random
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from app.analytics import train as train_mod
from app.analytics.service import AnalyticsService
from app.core.config import BACKEND_DIR, PROJECT_DIR, Settings
from app.telemetry.service import TelemetryService
from app.telemetry.simulator import SIMULATOR_PROTOCOL, SiteModel, load_config, read_metric
from app.telemetry.schemas import IngestMessage, IngestReading

logger = logging.getLogger(__name__)
STEP_S = 60
H = 3600
D = 86400

# (scenario, start offset s, duration s, expected equipment, expected metrics, kind)
SCENARIOS = [
    ("temperature_spike", 4 * D + 2 * H, 45 * 60, "SIM-RRU-001", {"temperature"}, "sudden"),
    ("power_anomaly", 4 * D + 8 * H, 60 * 60, "SIM-RRU-002", {"power_consumption"}, "sudden"),
    ("voltage_fluctuation", 4 * D + 14 * H, 30 * 60, "SIM-PSU-001", {"output_voltage"}, "sudden"),
    ("sensor_dropout", 4 * D + 18 * H, 20 * 60, "SIM-RRU-001", set(), "communication"),
    ("battery_degradation", 5 * D, 48 * H, "SIM-BAT-001", {"battery_voltage", "battery_temperature",
                                                          "state_of_health"}, "gradual"),
    ("psu_overheating", 5 * D + 12 * H, 36 * H, "SIM-PSU-001", {"temperature"}, "gradual"),
]
TOTAL = 7 * D
TRAIN_END = 3 * D
NORMAL_EVAL = (3 * D, 4 * D)
GRACE_S = 15 * 60
DROPOUT_SENSOR = "SIM-RRU-001-MON"


def _settings(tmp: Path) -> Settings:
    return Settings(_env_file=None, telemetry_sqlite_fallback=tmp / "validation.db", telemetry_watchdog_interval_s=3600,
                    upload_dir=tmp / "u", output_dir=tmp / "o", warmup=False)


def run(name: str, seed: int = 7, out_root: Path | None = None, total_s: int = TOTAL, log=print) -> dict:
    out = (out_root or PROJECT_DIR / "experiments" / "anomaly_validation") / name
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    tmp = Path(tempfile.mkdtemp(prefix="anomaly_validation_"))
    t_wall = time.perf_counter()
    tel = TelemetryService(_settings(tmp))
    tel.start()
    if not tel.started:
        raise RuntimeError(tel.error)
    tel.simulator = SimpleNamespace(running=True, backfilling=False, stop=lambda: None, status=lambda: {})
    ana = AnalyticsService(tel)
    ana.start(warm_start=False, worker=False)
    sim_cfg = load_config(BACKEND_DIR / "configs" / "telemetry" / "simulator.yaml")
    sim_cfg["events"]["sensor_dropouts_per_hour"] = 0  # communication failures only where scheduled
    for e in sim_cfg["equipment"]:
        for s in e["sensors"]:
            s["interval_s"] = STEP_S  # the harness samples once per simulated minute
    for s in sim_cfg["site_sensors"]:
        s["interval_s"] = STEP_S
    tel.register_simulated(sim_cfg)
    sensors = [(s, e) for e in sim_cfg["equipment"] for s in e["sensors"]] + [(s, None) for s in sim_cfg["site_sensors"]]

    t0 = (datetime.now(UTC) - timedelta(seconds=total_s + 3600)).replace(second=0, microsecond=0)
    model = SiteModel(sim_cfg, random.Random(seed), t0)
    rng = model.rng
    pending = sorted((t0 + timedelta(seconds=s), n, d) for n, s, d, *_ in SCENARIOS if s < total_s)
    timeline, health_trace = [], []
    trained, seq, outages, was_mains = False, 0, 0, True
    t = t0
    batch = []
    while t < t0 + timedelta(seconds=total_s):
        while pending and pending[0][0] <= t:
            _, n, d = pending.pop(0)
            if n != "sensor_dropout":
                model.trigger(n, d)
            timeline.append({"scenario": n, "start": t.isoformat(), "duration_s": d})
        model.step(t, STEP_S)
        drop = next((sc for sc in SCENARIOS if sc[0] == "sensor_dropout"), None)
        silent = drop and t0 + timedelta(seconds=drop[1]) <= t < t0 + timedelta(seconds=drop[1] + drop[2])
        for s, e in sensors:
            if silent and s["id"] == DROPOUT_SENSOR:
                continue
            readings = []
            for metric in s["metrics"]:
                v = read_metric(model.values, e, metric)
                if v is not None:
                    from app.telemetry.simulator import BOUNDS, NOISE

                    v = float(v) + rng.gauss(0, NOISE.get(metric, 0.0))
                    lo, hi = BOUNDS.get(metric, (None, None))
                    v = max(lo, v) if lo is not None else v
                    v = min(hi, v) if hi is not None else v
                readings.append(IngestReading(metric=metric, value=None if v is None else round(v, 3)))
            seq += 1
            batch.append(IngestMessage(sensor_id=s["id"], message_id=f"v{seq}", timestamp=t, readings=readings))
        if model.mains is False and was_mains:
            outages += 1
        was_mains = model.mains
        # one transaction per simulated minute: receive times (and therefore sensor status) stay exact
        tel.pipeline.ingest_many(batch, SIMULATOR_PROTOCOL, received_at=t)
        batch = []
        tel.check_status(now=t)
        t += timedelta(seconds=STEP_S)
        if int((t - t0).total_seconds()) % 1800 == 0:
            ana.tick(now=t)
            if True:
                health_trace.append({"t": t.isoformat(), **{eq: (ana.health.get(eq) or {}).get("status")
                                                            for eq in ("SIM-RRU-001", "SIM-RRU-002", "SIM-PSU-001",
                                                                       "SIM-BAT-001", "SIM-CAB-001")}})
        if not trained and t >= t0 + timedelta(seconds=TRAIN_END):
            if batch:
                tel.pipeline.ingest_many(batch, SIMULATOR_PROTOCOL, received_at=t)
                batch = []
            models = train_mod.train(tel.store.sessions, ana.store, ana.cfg, "simulated", t0, t, seed=seed,
                                     models_dir=tmp / "models")
            ana.engine.load_models(models, tmp)
            trained = True
            log(f"trained {len(models)} Isolation Forest models at day {(t - t0).total_seconds() / D:.1f}")
        if (t - t0).total_seconds() % D == 0:
            log(f"simulated day {(t - t0).total_seconds() / D:.0f} ({time.perf_counter() - t_wall:.0f} s)")
    if batch:
        tel.pipeline.ingest_many(batch, SIMULATOR_PROTOCOL, received_at=t)
    tel.check_status(now=t)
    ana.run_maintenance(now=t)
    ana.refresh_health(now=t)
    report = score(ana, tel, t0, t, timeline, health_trace)
    report["mains_outages"]["random_outages_in_period"] = outages
    report["runtime"] = {"wall_s": round(time.perf_counter() - t_wall, 1), "simulated_days": total_s / D,
                         "messages": seq, "engine": ana.engine.benchmark(), "seed": seed}
    ana.stop()
    tel.stop()
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(markdown(report), encoding="utf-8")
    report["path"] = str(out)
    return report


def score(ana: AnalyticsService, tel: TelemetryService, t0: datetime, end: datetime, timeline, health_trace) -> dict:
    _, anomalies = ana.store.anomalies_page(10000, 0)
    anomalies = sorted(anomalies, key=lambda a: a.first_detected_at)
    windows = []
    results = []
    for n, s, d, eq, metrics, kind in SCENARIOS:
        start, stop = t0 + timedelta(seconds=s), t0 + timedelta(seconds=s + d)
        if start >= end:
            continue
        windows.append((start, stop + timedelta(seconds=GRACE_S), eq))
        hits = [a for a in anomalies if a.equipment_id == eq and start <= a.first_detected_at <= stop +
                timedelta(seconds=GRACE_S) and (a.metric in metrics or a.kind == "multivariate")]
        r = {"scenario": n, "kind": kind, "equipment": eq, "start": start.isoformat(), "duration_min": d / 60}
        if kind == "communication":
            comm = [a for a in tel.alerts.store.alerts_page(1000, 0, sensor_id=DROPOUT_SENSOR)[1]
                    if a.rule_id == "sensor_communication" and start <= a.opened_at <= stop + timedelta(seconds=GRACE_S)]
            equipment_anoms = [a for a in anomalies if a.equipment_id == eq and start <= a.first_detected_at <= stop +
                               timedelta(seconds=GRACE_S)]
            r.update(communication_alert=bool(comm), communication_alert_cleared=bool(comm and comm[0].state == "cleared"),
                     equipment_anomalies_during_dropout=len(equipment_anoms),
                     passed=bool(comm) and not equipment_anoms)
        else:
            first = hits[0] if hits else None
            recs = [x for x in ana.store.recommendations_page(100, 0, equipment_id=eq, status=None)[1]
                    if x.metric in metrics]
            r.update(detected=bool(hits), anomalies=len(hits),
                     detectors=sorted({d_ for a in hits for d_ in a.detectors}),
                     delay_min=round((first.first_detected_at - start).total_seconds() / 60, 1) if first else None,
                     max_severity="critical" if any(a.severity == "critical" for a in hits) else (
                         "warning" if hits else None),
                     peak_score=max((a.peak_score for a in hits), default=None),
                     ended_after_recovery=all(not a.condition_active for a in hits) if hits and stop < end else None,
                     trend_recommendations=[{"key": x.condition_key, "evidence": x.evidence_level,
                                             "slope_per_day": x.evidence.get("slope_per_day")} for x in recs])
            r["passed"] = r["detected"] or bool(recs and kind == "gradual")
        results.append(r)

    def in_window(a):
        return any(ws <= a.first_detected_at <= we and a.equipment_id == eq for ws, we, eq in windows)

    warm = t0 + timedelta(hours=2)
    false_alarms = [a for a in anomalies if a.first_detected_at >= warm and not in_window(a)]
    normal_s = (TRAIN_END - 2 * H) + (NORMAL_EVAL[1] - NORMAL_EVAL[0])
    eval_fa = [a for a in false_alarms if t0 + timedelta(seconds=NORMAL_EVAL[0]) <= a.first_detected_at <
               t0 + timedelta(seconds=NORMAL_EVAL[1])]
    outages = [w for w in timeline if w["scenario"] == "mains_outage"]
    return {
        "disclaimer": "SIMULATED DATA ONLY: detection of simulator-injected patterns, not real-world accuracy.",
        "period": {"start": t0.isoformat(), "end": end.isoformat()},
        "scenarios": results,
        "false_alarms": {"total_outside_scenarios": len(false_alarms),
                         "per_day_all_normal_time": round(len(false_alarms) / ((end - t0).total_seconds() / D -
                                                                                sum(d for _, _, d, *_ in SCENARIOS) / D), 2),
                         "day3_normal_with_models": len(eval_fa),
                         "by_detector": _count(a.detector for a in false_alarms),
                         "by_metric": _count(a.metric or "multivariate" for a in false_alarms),
                         "by_severity": _count(a.severity for a in false_alarms),
                         "examples": [{"t": a.first_detected_at.isoformat(), "equipment": a.equipment_id,
                                       "metric": a.metric, "detector": a.detector, "severity": a.severity,
                                       "explanation": a.explanation} for a in false_alarms[:10]]},
        "mains_outages": {"random_outages_in_period": len(outages),
                          "context_suppressed_evaluations": ana.engine.stats["suppressed"]},
        "anomalies_total": len(anomalies),
        "severity_counts": _count(a.severity for a in anomalies),
        "health_at_end": {k: {"status": v["status"], "score": v["score"], "coverage": v["coverage"]}
                          for k, v in ana.health.items()},
        "health_trace": health_trace[::4],
        "recommendations": [{"equipment": r.equipment_id, "key": r.condition_key, "evidence": r.evidence_level,
                             "status": r.status, "condition": r.condition}
                            for r in ana.store.recommendations_page(200, 0, status=None)[1]],
        "normal_seconds": normal_s,
    }


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def markdown(r: dict) -> str:
    L = [f"# Anomaly detection validation ({r['runtime']['simulated_days']:.0f} simulated days)", "",
         f"> {r['disclaimer']}", "",
         f"Seed {r['runtime']['seed']} · {r['runtime']['messages']:,} messages · wall time {r['runtime']['wall_s']} s · "
         f"engine mean {r['runtime']['engine']['eval_ms_mean']} ms per evaluation "
         f"({r['runtime']['engine']['evaluations']:,} evaluations)", "",
         "## Injected scenarios", "",
         "| Scenario | Kind | Equipment | Detected | Delay (min) | Max severity | Detectors | Ended after recovery | "
         "Trend recommendation | Passed |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in r["scenarios"]:
        if s["kind"] == "communication":
            L.append(f"| {s['scenario']} | communication | {s['equipment']} | comm. alert: {s['communication_alert']} "
                     f"(cleared: {s['communication_alert_cleared']}) | — | — | — | — | equipment anomalies during "
                     f"dropout: {s['equipment_anomalies_during_dropout']} | {s['passed']} |")
        else:
            trend = ", ".join(f"{t['key']} ({t['evidence']})" for t in s["trend_recommendations"]) or "—"
            L.append(f"| {s['scenario']} | {s['kind']} | {s['equipment']} | {s['detected']} | {s['delay_min']} | "
                     f"{s['max_severity']} | {', '.join(s['detectors']) or '—'} | {s['ended_after_recovery']} | {trend} | "
                     f"{s['passed']} |")
    fa = r["false_alarms"]
    L += ["", "## Normal operation", "",
          f"- Anomalies outside the scenario windows (after 2 h warm-up): **{fa['total_outside_scenarios']}** "
          f"(≈ {fa['per_day_all_normal_time']} per simulated day across all equipment)",
          f"- Of these on day 3 (normal, with trained models active): {fa['day3_normal_with_models']}",
          f"- By detector: {fa['by_detector']} · by metric: {fa['by_metric']} · by severity: {fa['by_severity']}",
          f"- Random mains outages in the period: {r['mains_outages']['random_outages_in_period']} scheduled + random; "
          f"evaluations explained by the outage/recharge context: {r['mains_outages']['context_suppressed_evaluations']}",
          "", "## Health at the end", "", "| Equipment | Status | Score | Coverage |", "|---|---|---|---|"]
    for k, v in r["health_at_end"].items():
        L.append(f"| {k} | {v['status']} | {v['score']} | {v['coverage']} |")
    L += ["", "## Maintenance recommendations", ""]
    L += [f"- {x['equipment']} · {x['key']} · evidence {x['evidence']} · {x['status']}: {x['condition']}"
          for x in r["recommendations"]] or ["- none"]
    if fa["examples"]:
        L += ["", "## False-alarm examples", ""]
        L += [f"- {x['t']} {x['equipment']} {x['metric']} ({x['detector']}, {x['severity']}): {x['explanation']}"
              for x in fa["examples"]]
    return "\n".join(L) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--days", type=float, default=7.0)
    a = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    r = run(a.name, a.seed, total_s=int(a.days * D))
    print(markdown(r))
    print(f"Report: {r['path']}")


if __name__ == "__main__":
    main()
