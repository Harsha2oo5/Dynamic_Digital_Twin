"""Anomaly detection, health scoring and predictive maintenance (Milestone 3).

Synthetic series and temporary SQLite databases only; no hardware, no network.
"""

import copy
import math
import random
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.analytics import config as acfg
from app.analytics import detectors as det
from app.analytics import health as health_mod
from app.analytics import maintenance
from app.analytics.engine import AnalyticsEngine, mv_vector
from app.analytics.features import SeriesState
from app.main import create_app
from tests.conftest import make_settings

T0 = datetime(2026, 11, 1, tzinfo=UTC)
CFG, _ = acfg.load()


def cfg(**changes) -> dict:
    c = copy.deepcopy(CFG)
    for k, v in changes.items():
        c[k] = v
    return c


def feed(engine, values, metric="battery_temperature", category="battery", sensor="T-1", equipment="BAT-1",
         start=T0, step_s=60, source="physical"):
    for i, v in enumerate(values):
        if v is None:
            continue
        engine.on_reading(sensor, equipment, category, source, metric, v, start + timedelta(seconds=i * step_s))


def normal(n, base=30.0, amp=3.0, noise=0.2, seed=1):
    rng = random.Random(seed)
    return [base + amp * math.sin(2 * math.pi * i / 1440) + rng.gauss(0, noise) for i in range(n)]


# --- features -------------------------------------------------------------------------------------------
def test_irregular_samples_become_a_regular_grid_with_gaps():
    s = SeriesState(60, 30, 5, 3)
    times = [0, 10, 50, 61, 125, 400, 430]  # buckets 0,0,0,1,2,6,7
    for t, v in zip(times, [1, 2, 3, 10, 20, 30, 40]):
        s.add(T0 + timedelta(seconds=t), v)
    vals = list(s.values)
    assert vals[0] == pytest.approx(2.0) and vals[1] == 10 and vals[2] == 20
    assert all(math.isnan(v) for v in vals[3:6]) and vals[6] == 30
    assert s.add(T0 + timedelta(seconds=5), 99) == 0  # late reading into a closed bucket is ignored


def test_rate_of_change_never_spans_a_long_gap_and_forecast_resets():
    s = SeriesState(60, 60, 5, 2)
    for i in range(40):
        s.add(T0 + timedelta(minutes=i), 20.0)
    for i in range(50, 60):  # 10-minute silence, then data resumes at a different level
        s.add(T0 + timedelta(minutes=i), 35.0)
    s.add(T0 + timedelta(minutes=60), 35.0)
    f = s.features()
    assert s.level is not None and abs(s.level - 35.0) < 1.0  # forecast restarted after the gap
    s2 = SeriesState(60, 60, 5, 2)
    for i in range(40):
        s2.add(T0 + timedelta(minutes=i), 20.0)
    s2.add(T0 + timedelta(minutes=44), 35.0)
    s2.add(T0 + timedelta(minutes=45), 35.0)
    assert s2.features().roc is None  # the 10-min window crosses a 3-bucket gap
    assert f.roc is not None


def test_flagged_buckets_stay_out_of_the_baseline_until_they_dominate():
    s = SeriesState(60, 20, 3, 3, relearn_share=0.5)
    for i in range(15):
        s.add(T0 + timedelta(minutes=i), 10.0)
    for i in range(15, 21):
        s.add(T0 + timedelta(minutes=i), 50.0)
        s.flag_last()
    assert s.features().median == pytest.approx(10.0)
    for i in range(21, 40):
        s.add(T0 + timedelta(minutes=i), 50.0)
        s.flag_last()
    assert s.features().median == pytest.approx(50.0)  # the shift became the new normal


# --- detectors ------------------------------------------------------------------------------------------
def _features(history, value):
    s = SeriesState(60, len(history) + 1, 10, 5)
    for i, v in enumerate(history + [value, value]):
        s.add(T0 + timedelta(minutes=i), v)
    return s.features()


def test_robust_z_uses_the_scale_floor_on_flat_signals():
    f = _features([94.0] * 100, 94.2)
    r = det.robust_z(f, {"threshold": 6}, 1.0, "%", "State of health")
    assert r.score < 0.1  # 0.2 % on a perfectly flat signal is noise, not 1e9 sigmas
    r = det.robust_z(_features([94.0] * 100, 70.0), {"threshold": 6}, 1.0, "%", "State of health")
    assert r.score >= 2 and r.direction == "low" and "below" in r.detail


def test_detectors_return_none_without_enough_history():
    f = _features([1.0] * 5, 2.0)
    for name, fn in det.UNIVARIATE.items():
        assert fn(f, {"threshold": 6, "k": 3, "max_per_window": 5}, 0.1, "V", "x") is None, name


def test_iqr_rate_of_change_and_forecast_residual():
    rng = random.Random(3)
    hist = [50 + rng.gauss(0, 0.5) for _ in range(200)]
    f = _features(hist, 58.0)
    assert det.iqr(f, {"k": 3}, 0.1, "V", "v").score > 1
    assert det.rate_of_change(f, {"threshold": 6}, 0.1, "V", "v").score > 1
    assert det.forecast_residual(f, {"threshold": 6}, 0.1, "V", "v").score > 1
    g = _features(hist, 50.2)
    assert det.iqr(g, {"k": 3}, 0.1, "V", "v").score < 1 and det.robust_z(g, {"threshold": 6}, 0.1, "V", "v").score < 1


def test_transitions_detector():
    s = SeriesState(60, 60, 5, 5)
    for i in range(61):
        s.add(T0 + timedelta(minutes=i), float(i % 2 if i > 40 else 0))
    f = s.features(state_metric=True)
    assert det.transitions(f, {"max_per_window": 6}, 1, "state", "Door").score > 1


# --- configuration ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("mutate,needle", [
    (lambda c: c["categories"]["battery"]["metrics"].update(battery_voltage=["magic"]), "unknown detectors"),
    (lambda c: c["categories"]["battery"]["metrics"].update(warp_core=["robust_z"]), "unknown metric"),
    (lambda c: c["categories"]["equipment_cabinet"]["metrics"].update(door_open=["robust_z"]), "state metrics"),
    (lambda c: c["severity"].update(critical=0.5), "warning < critical"),
    (lambda c: c["health"]["bands"].update(warning=95), "health.bands"),
    (lambda c: c["categories"]["battery"].update(multivariate=["humidity"]), "multivariate"),
    (lambda c: c["maintenance"]["trends"]["battery"]["battery_voltage"].pop("action"), "inspection action"),
])
def test_config_validation(mutate, needle):
    c = copy.deepcopy(CFG)
    mutate(c)
    with pytest.raises(acfg.ConfigError, match=needle):
        acfg.validate(c)


def test_default_config_is_valid_and_hash_is_stable():
    acfg.validate(CFG)
    assert acfg.config_hash(CFG) == acfg.config_hash(copy.deepcopy(CFG))


# --- engine: real-time detection -------------------------------------------------------------------------
def test_normal_operation_is_not_flagged():
    e = AnalyticsEngine(cfg())
    feed(e, normal(2 * 1440))
    assert e.stats["opened"] == 0 and e.stats["evaluations"] > 2000


def test_injected_step_is_detected_with_severity_and_recovers():
    e = AnalyticsEngine(cfg())
    events = []
    e.publish = lambda kind, p: events.append(p["event"])
    vals = normal(600, amp=0) + [40.0] * 30 + normal(200, amp=0, seed=4)  # +10 degC for 30 min, then normal
    feed(e, vals)
    opened = [ep.anomaly for ep in e.episodes.values()]
    assert len(opened) == 1
    a = opened[0]
    assert a.severity == "critical" and a.metric == "battery_temperature" and a.status == "resolved"
    assert not a.condition_active and a.resolved_by == "system" and "robust" in a.explanation or a.explanation
    assert {"robust_z"} <= set(a.detectors)
    assert events[0] == "anomaly_opened" and events[-1] == "anomaly_ended"
    assert a.first_detected_at - (T0 + timedelta(minutes=600)) <= timedelta(minutes=3)


def test_moderate_deviation_is_a_warning_and_single_blips_are_ignored():
    e = AnalyticsEngine(cfg())
    vals = normal(800, amp=0)
    for i in range(600, 800):  # slow ramp to +4.5 degC over an hour, then held: moderate, not sudden
        vals[i] += min(4.5, (i - 600) * 4.5 / 60)
    feed(e, vals)
    sev = {ep.anomaly.severity for ep in e.episodes.values()}
    assert sev == {"warning"}
    e2 = AnalyticsEngine(cfg())
    blip = normal(600, amp=0)
    blip[500] = 45.0  # one bucket only: below min_consecutive
    feed(e2, blip)
    assert e2.stats["opened"] == 0


def test_recurrence_within_cooldown_reopens_instead_of_duplicating():
    e = AnalyticsEngine(cfg())
    vals = normal(600, amp=0) + [42.0] * 10 + normal(25, amp=0, seed=6) + [42.0] * 10 + normal(60, amp=0, seed=7)
    feed(e, vals)
    assert len(e.episodes) == 1
    a = next(iter(e.episodes.values())).anomaly
    assert a.occurrences == 2 and e.stats["reopened"] == 1


def test_sensor_gap_is_not_an_equipment_anomaly():
    e = AnalyticsEngine(cfg())
    vals = normal(600) + [None] * 40 + [v + 4 for v in normal(120, seed=8)]  # silent 40 min, resumes slightly shifted
    feed(e, vals)
    assert e.stats["opened"] == 0


def test_mains_outage_context_suppresses_expected_discharge():
    e = AnalyticsEngine(cfg())
    rng = random.Random(1)
    for i in range(900):
        t = T0 + timedelta(minutes=i)
        outage = 600 <= i < 660
        e.on_reading("PWR", None, "site", "physical", "mains_available", 0.0 if outage else 1.0, t)
        current = -22.0 if outage else (15.0 * math.exp(-(i - 660) / 60) if i >= 660 else 0.3)
        e.on_reading("I-1", "BAT-1", "battery", "physical", "battery_current", current + rng.gauss(0, 0.1), t)
    assert e.stats["opened"] == 0 and e.stats["suppressed"] > 0
    # the same current pattern without any mains sensor is flagged
    e2 = AnalyticsEngine(cfg())
    for i in range(700):
        current = -22.0 if 600 <= i < 660 else 0.3
        e2.on_reading("I-1", "BAT-1", "battery", "physical", "battery_current", current, T0 + timedelta(minutes=i))
    assert e2.stats["opened"] == 1


def test_isolation_forest_scores_unusual_combinations_and_respects_data_source(tmp_path):
    from app.analytics import train as train_mod

    rows = []
    sink = lambda c, s, eq, t, names, x: rows.append((names, x))  # noqa: E731
    e = AnalyticsEngine(cfg(), mv_sink=sink)
    rng = random.Random(2)
    for i in range(1500):
        t = T0 + timedelta(minutes=i)
        traffic = 0.5 + 0.4 * math.sin(2 * math.pi * i / 1440)
        e.on_reading("R", "RRU-1", "radio_unit", "simulated", "power_consumption", 200 + 300 * traffic + rng.gauss(0, 3), t)
        e.on_reading("R", "RRU-1", "radio_unit", "simulated", "temperature", 35 + 15 * traffic + rng.gauss(0, 0.3), t)
    X = np.vstack([x for _, x in rows])
    names = rows[0][0]
    assert names == ["temperature:value", "power_consumption:value"]
    r = train_mod.fit(X, 0.999, seed=1)
    assert r["evaluation"]["holdout_flagged_rate"] < 0.05
    import joblib

    path = tmp_path / "m.joblib"
    joblib.dump({"model": r["model"], "features": names, "median": r["median"], "scale": r["scale"],
                 "threshold": r["threshold"], "center": r["center"]}, path)
    row = SimpleNamespace(id=1, category="radio_unit", data_source="simulated", artifact_path=str(path))
    live = AnalyticsEngine(cfg())
    assert live.load_models([row], tmp_path) == ["radio_unit/simulated#1"]
    for i in range(720):  # normal pairs, then high-traffic temperature with low-traffic power: each value is
        t = T0 + timedelta(minutes=i)  # inside its normal range, only the combination is unusual
        traffic = 0.5 + 0.4 * math.sin(2 * math.pi * i / 1440)
        odd = i >= 650
        live.on_reading("R", "RRU-1", "radio_unit", "simulated", "power_consumption",
                        (240 if odd else 200 + 300 * traffic) + rng.gauss(0, 3), t)
        live.on_reading("R", "RRU-1", "radio_unit", "simulated", "temperature",
                        (48 if odd else 35 + 15 * traffic) + rng.gauss(0, 0.3), t)
    mv = [ep.anomaly for k, ep in live.episodes.items() if k[3] == "multivariate"]
    assert live.stats["mv_scored"] > 0 and mv and "Unusual combination" in mv[0].explanation
    phys = AnalyticsEngine(cfg())
    phys.load_models([row], tmp_path)
    feed(phys, [400.0] * 200, metric="power_consumption", category="radio_unit", sensor="R", equipment="RRU-1")
    assert phys.stats["mv_scored"] == 0  # a simulated-data model never scores physical equipment


def test_mv_vector_requires_every_feature():
    assert mv_vector({"a": SimpleNamespace(value=None)}, ["a"]) is None
    assert mv_vector({"a": SimpleNamespace(value=1.0)}, ["a", "b"]) is None


# --- health -----------------------------------------------------------------------------------------------------
def anomaly(sev="warning", active=True, hours_ago=0.0, duration_h=0.5, occ=1, aid=1, metric="temperature"):
    now = T0 + timedelta(days=1)
    first = now - timedelta(hours=hours_ago + duration_h)
    return SimpleNamespace(id=aid, severity=sev, condition_active=active, first_detected_at=first,
                           last_detected_at=now - timedelta(hours=hours_ago),
                           ended_at=None if active else now - timedelta(hours=hours_ago), occurrences=occ,
                           metric=metric, detector="robust_z")


COV = {"coverage": 1.0, "provided": ["a", "b"], "usable": ["a", "b"], "detail": {}}


def test_health_score_bands_and_explanation():
    now = T0 + timedelta(days=1)
    r = health_mod.compute("E", "battery", COV, [], [], [], 1.0, CFG, now)
    assert (r["score"], r["status"]) == (100.0, "healthy")
    r = health_mod.compute("E", "battery", COV, [anomaly("critical", duration_h=2)], [], [], 1.0, CFG, now)
    assert r["status"] in ("warning", "degraded") and r["components"][0]["type"] == "anomaly"
    many = [anomaly("critical", aid=i, duration_h=3, occ=3) for i in range(4)]
    assert health_mod.compute("E", "battery", COV, many, [], [], 1.0, CFG, now)["status"] == "critical"


def test_old_anomalies_decay_and_communication_alerts_do_not_count():
    now = T0 + timedelta(days=1)
    fresh = health_mod.anomaly_penalty(anomaly("critical", active=False, hours_ago=0.1), now, CFG["health"])
    old = health_mod.anomaly_penalty(anomaly("critical", active=False, hours_ago=12), now, CFG["health"])
    gone = health_mod.anomaly_penalty(anomaly("critical", active=False, hours_ago=30), now, CFG["health"])
    assert fresh > old > 0 and gone == 0
    comm = SimpleNamespace(id=9, rule_id="sensor_communication", severity="critical")
    r = health_mod.compute("E", "battery", COV, [], [comm], [], 0.2, CFG, now)
    assert r["score"] == 100.0 and r["data_quality"] == 0.2  # sensor reliability is reported separately


def test_insufficient_data_gives_unknown_not_a_score():
    now = T0
    low = {"coverage": 0.25, "provided": ["a", "b", "c", "d"], "usable": ["a"], "detail": {}}
    r = health_mod.compute("E", "battery", low, [anomaly()], [], [], 1.0, CFG, now)
    assert r["score"] is None and r["status"] == "unknown" and "insufficient data" in r["unknown_reason"]
    none = {"coverage": 0.0, "provided": [], "usable": [], "detail": {}}
    assert health_mod.compute("E", "antenna", none, [], [], [], None, CFG, now)["status"] == "unknown"


# --- maintenance -----------------------------------------------------------------------------------------------
def _rows(values, start=T0, step_h=1.0, per_hour=4, noise=0.05, seed=1):
    rng = random.Random(seed)
    out = []
    for i, v in enumerate(values):
        for k in range(per_hour):
            out.append(("S", "EQ", start + timedelta(hours=i * step_h, minutes=15 * k), v + rng.gauss(0, noise), "physical"))
    return out


def test_trend_detection_and_evidence_levels():
    hours = 7 * 24
    rising = [30 + 0.04 * h + 1.5 * math.sin(2 * math.pi * h / 24) for h in range(hours)]  # +0.96 degC/day
    h, v = maintenance.hourly_medians(_rows(rising))
    t = maintenance.trend(h, v)
    assert 0.8 < t["slope_per_day"] < 1.1 and t["p_value"] < 0.001 and maintenance.evidence_level(t) == "high"
    flat = [30 + 1.5 * math.sin(2 * math.pi * h / 24) for h in range(hours)]
    t2 = maintenance.trend(*maintenance.hourly_medians(_rows(flat, noise=0.3)))
    assert abs(t2["slope_per_day"]) < 0.2


def test_analyse_equipment_recommends_only_real_degradation():
    now = T0 + timedelta(days=7)
    eq = SimpleNamespace(equipment_id="BAT-1", category="battery", data_source="physical")
    rising = _rows([30 + 0.05 * h for h in range(7 * 24)])
    series = lambda metric, s, e, equipment_id=None: rising if metric == "battery_temperature" else []  # noqa: E731
    cands, assessed = maintenance.analyse_equipment(eq, {"battery_temperature"}, series, CFG, now, [], [], [])
    assert assessed and len(cands) == 1
    c = cands[0]
    assert c["condition_key"] == "trend:battery_temperature:increasing" and c["evidence_level"] == "high"
    assert "no failure date" in c["condition"] and c["evidence"]["series"] and c["action"]
    short = _rows([30 + 0.05 * h for h in range(24)])
    cands, assessed = maintenance.analyse_equipment(eq, {"battery_temperature"}, lambda *a, **k: short, CFG, now,
                                                   [], [], [])
    assert cands == [] and not assessed  # too little history: no recommendation, nothing resolved


def test_outage_hours_are_excluded_for_mains_dependent_trends():
    mains = [("P", None, T0 + timedelta(hours=10, minutes=5), 0.0, "physical")]
    excl = maintenance.outage_hours(mains, 3 * 3600)
    base = int((T0 + timedelta(hours=10)).timestamp() // 3600)
    assert {base, base + 1, base + 2, base + 3} <= excl


def test_anomaly_frequency_recommendation():
    now = T0 + timedelta(days=7)
    eq = SimpleNamespace(equipment_id="RRU-1", category="radio_unit", data_source="physical")
    times = [now - timedelta(days=5)] + [now - timedelta(hours=h) for h in (1, 3, 5, 7)]
    cands, _ = maintenance.analyse_equipment(eq, set(), lambda *a, **k: [], CFG, now, [], times, [])
    assert [c["condition_key"] for c in cands] == ["anomaly_frequency"]


# --- service + API ------------------------------------------------------------------------------------------------
@pytest.fixture
def client(tmp_path):
    s = make_settings(tmp_path, telemetry_ingest_api_key=SecretStr("k"), telemetry_watchdog_interval_s=3600)
    app = create_app(s)
    with TestClient(app) as c:
        yield c


def _setup(client, minutes=180, jump_at=None, jump=12.0, end_ago_min=1):
    client.post("/api/telemetry/equipment", json={"equipment_id": "BAT-001", "category": "battery", "name": "B"})
    client.post("/api/telemetry/sensors", json={"sensor_id": "BT-1", "name": "t", "sensor_type": "temperature",
                                                "protocol": "http", "metrics": ["battery_temperature", "battery_voltage"],
                                                "equipment_id": "BAT-001", "sampling_interval_s": 60})
    start = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=minutes + end_ago_min - 1)
    rng = random.Random(1)
    msgs = []
    for i in range(minutes):
        temp = 28 + rng.gauss(0, 0.2) + (jump if jump_at is not None and i >= jump_at else 0)
        msgs.append({"sensor_id": "BT-1", "timestamp": (start + timedelta(minutes=i)).isoformat(),
                     "readings": [{"metric": "battery_temperature", "value": round(temp, 3)},
                                  {"metric": "battery_voltage", "value": round(53.5 + rng.gauss(0, 0.03), 3)}]})
    r = client.post("/api/telemetry/ingest/batch", json={"messages": msgs}, headers={"X-API-Key": "k"})
    assert r.json()["stored"] == 2 * minutes
    return start + timedelta(minutes=minutes - 1)


def test_api_anomaly_lifecycle_health_and_details(client):
    _setup(client, jump_at=150)
    assert client.get("/api/analytics/status").json()["available"]
    page = client.get("/api/analytics/anomalies", params={"equipment_id": "BAT-001"}).json()
    assert page["total"] == 1
    a = page["items"][0]
    assert a["metric"] == "battery_temperature" and a["severity"] == "critical" and a["condition_active"]
    assert a["expected"] == pytest.approx(28, abs=0.5) and a["evidence"]["findings"] and a["data_source"] == "physical"
    client.post("/api/analytics/maintenance/run")
    h = client.get("/api/analytics/health/equipment/BAT-001").json()
    assert h["status"] in ("warning", "degraded", "critical") and h["score"] < 100 and h["coverage"] == 1.0
    assert h["components"][0]["type"] == "anomaly"
    assert client.post(f"/api/analytics/anomalies/{a['id']}/ack", json={"by": "harsha"}).json()["status"] == "acknowledged"
    r = client.post(f"/api/analytics/anomalies/{a['id']}/resolve", json={"by": "harsha", "note": "fan replaced"}).json()
    assert r["status"] == "resolved" and r["resolution_note"] == "fan replaced"
    assert client.post(f"/api/analytics/anomalies/{a['id']}/ack", json={"by": "x"}).status_code == 409
    assert client.post("/api/analytics/anomalies/999/resolve", json={"by": "x"}).status_code == 404
    d = client.get("/api/analytics/equipment/BAT-001/details").json()
    assert d["equipment"]["equipment_id"] == "BAT-001" and d["anomalies"] and d["health"]["status"]
    assert "visual" in d and d["sensors"][0]["sensor_id"] == "BT-1"
    s = client.get("/api/analytics/health/summary").json()
    assert s["equipment"] == 1 and s["active_anomalies"] == 1
    assert client.get("/api/health").json()["status"] == "ok"  # vision unaffected


def test_api_unknown_health_when_data_insufficient(client):
    client.post("/api/telemetry/equipment", json={"equipment_id": "RTR-001", "category": "router", "name": "R"})
    client.post("/api/analytics/maintenance/run")
    h = client.get("/api/analytics/health/equipment/RTR-001").json()
    assert h["status"] == "unknown" and h["score"] is None


def test_api_config_versioning_and_validation(client):
    c = client.get("/api/analytics/config").json()
    assert c["version"] == 1 and "grid_s" in c["content"]
    bad = client.put("/api/analytics/config", json={"content": "grid_s: 60\n", "by": "harsha"})
    assert bad.status_code == 422
    new = c["content"].replace("min_consecutive: 2", "min_consecutive: 3")
    r = client.put("/api/analytics/config", json={"content": new, "by": "harsha", "note": "fewer blips"})
    assert r.status_code == 200 and r.json()["version"] == 2
    versions = client.get("/api/analytics/config/versions").json()
    assert [v["version"] for v in versions] == [2, 1] and versions[0]["active"]
    assert client.get("/api/analytics/config").json()["parsed"]["episode"]["min_consecutive"] == 3
    assert client.get("/api/analytics/models").json() == []


def test_websocket_receives_anomaly_events(client):
    last = _setup(client, minutes=150, end_ago_min=10)
    with client.websocket_connect("/ws/telemetry") as ws:
        assert ws.receive_json()["type"] == "snapshot"
        for k in range(1, 6):  # five new minutes at 45 degC, after the stored history
            r = client.post("/api/telemetry/ingest", headers={"X-API-Key": "k"}, json={
                "sensor_id": "BT-1", "timestamp": (last + timedelta(minutes=k)).isoformat(),
                "readings": [{"metric": "battery_temperature", "value": 45.0}]})
            assert r.json()["good"] == 1
        # the anomaly exists, so its event is already queued for this socket (no blocking wait)
        assert client.get("/api/analytics/anomalies").json()["total"] == 1
        events = []
        while not events:
            msg = ws.receive_json()
            events += [e for e in msg.get("events", []) if e["type"] == "anomaly"]
        assert events[0]["anomaly"]["metric"] == "battery_temperature" and events[0]["event"] == "anomaly_opened"


def test_anomalies_and_open_episodes_survive_restart(tmp_path):
    s = make_settings(tmp_path, telemetry_ingest_api_key=SecretStr("k"), telemetry_watchdog_interval_s=3600)
    with TestClient(create_app(s)) as c:
        _setup(c, jump_at=150)
        aid = c.get("/api/analytics/anomalies").json()["items"][0]["id"]
    with TestClient(create_app(s)) as c:
        engine = c.app.state.analytics.engine
        assert any(ep.anomaly.id == aid for ep in engine.episodes.values())  # no duplicate after restart
        assert engine.series  # feature windows warm-started from stored history
        assert c.get("/api/analytics/anomalies").json()["total"] == 1


def test_simulator_fault_scenarios_change_equipment_behaviour():
    from app.core.config import BACKEND_DIR
    from app.telemetry.simulator import SiteModel, load_config

    sim = load_config(BACKEND_DIR / "configs" / "telemetry" / "simulator.yaml")
    sim["events"] = {}
    m = SiteModel(sim, random.Random(1), T0)
    base = m.step(T0, 60)
    t_rru, p_rru = base["rru"][1]["temperature"], base["rru"][2]["power_consumption"]
    m.trigger("temperature_spike", 600)
    m.trigger("power_anomaly", 600)
    m.trigger("battery_degradation", 3600)
    v = m.step(T0 + timedelta(minutes=1), 60)
    assert v["rru"][1]["temperature"] > t_rru + 15 and v["rru"][2]["power_consumption"] > p_rru + 200
    late = m.step(T0 + timedelta(minutes=59), 60)
    assert late["battery_voltage"] < v["battery_voltage"] - 2  # gradual
    after = m.step(T0 + timedelta(hours=2), 60)
    assert after["rru"][1]["temperature"] < t_rru + 5  # recovered


def test_outage_only_metrics_are_judged_again_as_soon_as_mains_returns():
    e = AnalyticsEngine(cfg())
    t_restore = T0 + timedelta(minutes=600)
    e.on_reading("PWR", None, "site", "physical", "mains_available", 1.0, T0)
    e.on_reading("PWR", None, "site", "physical", "mains_available", 0.0, T0 + timedelta(minutes=540))
    e.on_reading("I", "BAT-1", "battery", "physical", "battery_current", 20.0, T0 + timedelta(minutes=540))
    assert e.suppressed_by_context("power_supply_unit", "output_voltage", T0 + timedelta(minutes=560))  # outage
    e.on_reading("PWR", None, "site", "physical", "mains_available", 1.0, t_restore)
    later = t_restore + timedelta(minutes=45)
    assert e.suppressed_by_context("power_supply_unit", "output_voltage", later) is None  # rectifier back
    assert e.suppressed_by_context("battery", "battery_current", later)  # still charging (20 A)
    e.on_reading("I", "BAT-1", "battery", "physical", "battery_current", 0.3, later)  # charge complete
    assert e.suppressed_by_context("battery", "battery_current", later + timedelta(minutes=30))  # settling
    assert e.suppressed_by_context("battery", "battery_current", later + timedelta(minutes=90)) is None


def test_detector_overrides_apply_per_metric():
    assert acfg.detector_params(CFG, "robust_z", "power_consumption")["threshold"] == 10.0
    assert acfg.detector_params(CFG, "robust_z", "battery_voltage")["threshold"] == 6.0


def test_no_forecast_false_alarm_when_voltage_returns_after_an_outage():
    """PSU output collapses to 0 V during an outage (expected) and returns to float voltage: no anomaly."""
    e = AnalyticsEngine(cfg())
    rng = random.Random(4)
    for i in range(900):
        t = T0 + timedelta(minutes=i)
        outage = 600 <= i < 640
        e.on_reading("PWR", None, "site", "physical", "mains_available", 0.0 if outage else 1.0, t)
        e.on_reading("P", "PSU-1", "power_supply_unit", "physical", "output_voltage",
                     0.0 if outage else 53.5 + rng.gauss(0, 0.03), t)
    assert e.stats["opened"] == 0


def test_oscillation_shortly_after_an_outage_is_detected():
    e = AnalyticsEngine(cfg())
    rng = random.Random(5)
    for i in range(800):
        t = T0 + timedelta(minutes=i)
        outage = 600 <= i < 640
        osc = 700 <= i < 730
        v = 0.0 if outage else 53.5 + rng.gauss(0, 0.03) + ((1.1 if i % 2 else -1.1) if osc else 0.0)
        e.on_reading("PWR", None, "site", "physical", "mains_available", 0.0 if outage else 1.0, t)
        e.on_reading("P", "PSU-1", "power_supply_unit", "physical", "output_voltage", v, t)
    opened = [ep.anomaly for ep in e.episodes.values()]
    assert len(opened) == 1 and opened[0].first_detected_at >= T0 + timedelta(minutes=700)
