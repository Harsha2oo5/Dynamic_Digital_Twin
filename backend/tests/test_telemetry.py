"""Sensor integration and telemetry: registry, ingestion, storage, status, alerts, MQTT, simulator, API, WebSocket.

Everything runs against temporary SQLite databases; MQTT uses a fake client (no broker), and no hardware is assumed.
"""

import json
import random
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from app.main import create_app
from app.telemetry import catalog, db
from app.telemetry.adapters import MqttAdapter, parse_topic
from app.telemetry.alerts import AlertConfig, AlertEngine, Rule, load_rules
from app.telemetry.schemas import IngestMessage, IngestReading, SensorCreate, SensorUpdate
from app.telemetry.service import TelemetryService, bucket_series
from app.telemetry.simulator import SiteModel, Simulator, SimulatorError, load_config
from tests.conftest import make_settings

NOW = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)


def tsettings(tmp_path, **kw):
    kw.setdefault("telemetry_sqlite_fallback", tmp_path / "telemetry.db")
    kw.setdefault("telemetry_watchdog_interval_s", 3600)  # tests drive check_status() themselves
    return make_settings(tmp_path, **kw)


@pytest.fixture
def svc(tmp_path):
    s = TelemetryService(tsettings(tmp_path))
    s.start()
    assert s.started, s.error
    yield s
    s.stop()


def add_battery(svc, sensor_id="VOLT-001", metrics=("battery_voltage", "battery_temperature"), interval=10.0,
                protocol="http"):
    if svc.store.get_equipment("BAT-001") is None:
        svc.store.create_equipment(svc.site_id, {"equipment_id": "BAT-001", "category": "battery", "name": "Battery"})
    body = SensorCreate(sensor_id=sensor_id, name="v", sensor_type="voltage", protocol=protocol, metrics=list(metrics),
                        equipment_id="BAT-001", sampling_interval_s=interval)
    svc.store.create_sensor(svc.site_id, body.model_dump())
    svc.sensor_changed(sensor_id)


def msg(sensor_id="VOLT-001", ts=NOW, mid=None, **readings):
    return IngestMessage(sensor_id=sensor_id, message_id=mid, timestamp=ts,
                         readings=[IngestReading(metric=k, value=v[0] if isinstance(v, tuple) else v,
                                                 unit=v[1] if isinstance(v, tuple) else None)
                                   for k, v in readings.items()])


# --- schema, catalogue, migrations ---------------------------------------------------------------------
def test_migration_matches_models(tmp_path):
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from app.telemetry.models import Base

    engine = db.make_engine(f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    db.upgrade(engine)
    with engine.connect() as c:
        assert compare_metadata(MigrationContext.configure(c), Base.metadata) == []
    from alembic.script import ScriptDirectory

    head = ScriptDirectory(str(db.MIGRATIONS_DIR)).get_current_head()
    assert db.current_revision(engine) == head  # migrated to the latest revision (0002 adds the analytics tables)


@pytest.mark.parametrize("metric,value,unit,expected", [
    ("battery_voltage", 52100, "mV", 52.1), ("battery_temperature", 104, "°F", 40.0),
    ("temperature", 300.15, "K", 27.0), ("state_of_charge", 0.5, "ratio", 50.0), ("output_current", 2500, "mA", 2.5),
    ("battery_voltage", 52.1, None, 52.1), ("vswr", 1.3, None, 1.3),
])
def test_unit_normalisation(metric, value, unit, expected):
    assert catalog.normalise(metric, value, unit)[0] == pytest.approx(expected)


def test_incompatible_units_and_plausibility():
    with pytest.raises(catalog.UnitError):
        catalog.normalise("battery_voltage", 3, "A")
    with pytest.raises(catalog.UnitError):
        catalog.normalise("battery_voltage", 3, "furlongs")
    assert not catalog.plausible("battery_voltage", 400)
    assert not catalog.plausible("door_open", 0.5) and catalog.plausible("door_open", 1)


def test_schema_validation():
    with pytest.raises(ValidationError, match="secret"):
        SensorCreate(sensor_id="S-1", name="s", sensor_type="voltage", protocol="mqtt", metrics=["battery_voltage"],
                     category="battery", config={"mqtt": {"password": "hunter2"}})
    SensorCreate(sensor_id="S-1", name="s", sensor_type="voltage", protocol="mqtt", metrics=["battery_voltage"],
                 category="battery", config={"password_env": "MQTT_PASSWORD"})
    with pytest.raises(ValidationError, match="unknown metrics"):
        SensorUpdate(metrics=["warp_core_temperature"])
    with pytest.raises(ValidationError, match="timezone"):
        IngestMessage(sensor_id="S-1", timestamp=datetime(2026, 1, 1), readings=[{"metric": "humidity", "value": 1}])
    with pytest.raises(ValidationError):
        SensorCreate(sensor_id="S 1!", name="s", sensor_type="voltage", protocol="http", metrics=["humidity"],
                     category="site")


# --- registry ---------------------------------------------------------------------------------------
def test_registration_and_association(svc):
    add_battery(svc)
    s = svc.store.get_sensor("VOLT-001")
    assert s.category == "battery" and s.equipment_id == "BAT-001" and s.data_source == "physical"
    from app.telemetry.store import Conflict, NotFound

    with pytest.raises(Conflict):
        add_battery(svc)
    with pytest.raises(NotFound):
        svc.store.associate("VOLT-001", "NOPE-1")
    svc.store.create_equipment(svc.site_id, {"equipment_id": "BAT-002", "category": "battery", "name": "B2"})
    svc.pipeline.ingest(msg(battery_voltage=52.0), "http")
    svc.store.associate("VOLT-001", "BAT-002")
    svc.sensor_changed("VOLT-001")
    svc.pipeline.ingest(msg(ts=NOW + timedelta(seconds=10), battery_voltage=52.1), "http")
    _, rows = svc.store.measurements_page(10, 0, sensor_id="VOLT-001")
    assert sorted(r.equipment_id for r in rows) == ["BAT-001", "BAT-002"]  # history keeps the old association


def test_multiple_sensors_per_equipment(svc):
    add_battery(svc, "VOLT-001", ("battery_voltage",))
    add_battery(svc, "TEMP-001", ("battery_temperature",))
    eq = svc.equipment_out(svc.store.get_equipment("BAT-001"))
    assert eq["sensor_ids"] == ["TEMP-001", "VOLT-001"]


# --- ingestion ----------------------------------------------------------------------------------------
def test_unregistered_and_wrong_channel_are_rejected(svc):
    r = svc.pipeline.ingest(msg("GHOST-1", battery_voltage=52), "http")
    assert r.rejected == 1 and r.stored == 0 and "not registered" in r.errors[0]
    assert svc.store.get_sensor("GHOST-1") is None and svc.store.list_equipment() == []
    add_battery(svc)
    assert svc.pipeline.ingest(msg(battery_voltage=52), "mqtt").rejected == 1  # registered for http
    assert svc.pipeline.ingest(msg(battery_voltage=52), "simulator").rejected == 1  # physical, not simulated
    r = svc.pipeline.ingest(msg(battery_voltage=52, humidity=40), "http")
    assert r.stored == 1 and r.rejected == 1 and "not configured" in r.errors[0]


def test_normalisation_missing_and_invalid(svc):
    add_battery(svc)
    r = svc.pipeline.ingest(msg(battery_voltage=(52100, "mV"), battery_temperature=None), "http")
    assert (r.good, r.missing) == (1, 1)
    latest = {l["metric"]: l for l in svc.latest()}
    assert latest["battery_voltage"]["value"] == pytest.approx(52.1) and latest["battery_voltage"]["unit"] == "V"
    assert latest["battery_temperature"]["value"] is None and latest["battery_temperature"]["quality"] == "missing"
    r = svc.pipeline.ingest(msg(ts=NOW + timedelta(seconds=5), battery_voltage=(999, "V"),
                                battery_temperature=(3, "A")), "http")
    assert r.invalid == 2
    _, rows = svc.store.measurements_page(10, 0, quality="invalid")
    assert all(m.value is None and m.raw_value for m in rows)
    assert not any(l["current"] for l in svc.latest())  # invalid readings are never shown as current values


def test_duplicates_by_message_id_timestamp_and_database(svc, tmp_path):
    add_battery(svc)
    assert svc.pipeline.ingest(msg(mid="m1", battery_voltage=52), "http").stored == 1
    assert svc.pipeline.ingest(msg(mid="m1", battery_voltage=52), "http").duplicates == 1
    assert svc.pipeline.ingest(msg(mid="m2", battery_voltage=52), "http").duplicates == 1  # same timestamp
    # a fresh pipeline (e.g. after a restart, cold caches) is still protected by the unique constraint
    from app.telemetry.ingest import IngestionPipeline

    p2 = IngestionPipeline(svc.store, None)
    p2.sensors = dict(svc.pipeline.sensors)
    r = p2.ingest(msg(mid="m3", battery_voltage=52), "http")
    assert r.duplicates == 1 and r.stored == 0
    assert svc.store.counts()["measurements"] == 1


def test_out_of_order_is_history_only(svc):
    add_battery(svc)
    svc.alerts.config = AlertConfig([Rule("low", "battery_voltage", "below", warning=47)])
    svc.pipeline.ingest(msg(ts=NOW, battery_voltage=52), "http")
    r = svc.pipeline.ingest(msg(ts=NOW - timedelta(minutes=5), battery_voltage=40), "http")
    assert r.out_of_order == 1 and r.stored == 1
    assert svc.latest()[0]["value"] == 52  # latest unchanged
    assert svc.alerts.active() == []  # late readings never raise alerts
    _, rows = svc.store.measurements_page(10, 0, quality="out_of_order")
    assert rows[0].value == 40 and "older than latest" in rows[0].quality_note


def test_future_timestamps_are_rejected(svc):
    add_battery(svc)
    r = svc.pipeline.ingest(msg(ts=datetime.now(UTC) + timedelta(hours=1), battery_voltage=52), "http")
    assert r.rejected == 1 and "future" in r.errors[0]


# --- connection status, stale data, recovery ----------------------------------------------------------
def test_stale_offline_alert_and_recovery(svc):
    add_battery(svc, interval=10)  # stale after 30 s, offline after 100 s
    t0 = datetime.now(UTC) - timedelta(minutes=10)
    svc.pipeline.ingest(msg(ts=t0, battery_voltage=52), "http", received_at=t0)
    svc.check_status(t0 + timedelta(seconds=5))
    assert svc.sensor_out(svc.store.get_sensor("VOLT-001"), t0 + timedelta(seconds=5))["status"] == "online"
    svc.check_status(t0 + timedelta(seconds=60))
    assert svc._status["VOLT-001"][0] == "stale"
    svc.check_status(t0 + timedelta(seconds=200))
    assert svc._status["VOLT-001"][0] == "offline"
    comm = [a for a in svc.alerts.active() if a.rule_id == "sensor_communication"]
    assert len(comm) == 1 and comm[0].severity == "warning" and "Requires investigation" in comm[0].message
    svc.check_status(t0 + timedelta(seconds=1200))  # offline longer than critical_after_s (900 s)
    assert [a.severity for a in svc.alerts.active()] == ["critical"]
    last = svc.latest_out(svc.pipeline.latest[("VOLT-001", "battery_voltage")], t0 + timedelta(seconds=1200))
    assert last["value"] == 52 and last["current"] is False and last["sensor_status"] == "offline"
    t1 = t0 + timedelta(seconds=1210)
    svc.pipeline.ingest(msg(ts=t1, battery_voltage=52.2), "http", received_at=t1)
    svc.check_status(t1 + timedelta(seconds=1))
    assert svc._status["VOLT-001"][0] == "online" and svc.alerts.active() == []
    events = [e.status for e in svc.store.status_events(t0 - timedelta(seconds=1), "VOLT-001")]
    assert events == ["online", "stale", "offline", "online"]
    av = {a["sensor_id"]: a for a in svc.availability(t0, t1 + timedelta(seconds=1))}
    assert 0 < av["VOLT-001"]["online_share"] < 0.2


def test_never_seen_and_disabled_sensors(svc):
    add_battery(svc)
    assert svc.sensor_out(svc.store.get_sensor("VOLT-001"))["status"] == "unknown"
    svc.store.update_sensor("VOLT-001", {"enabled": False})
    svc.sensor_changed("VOLT-001")
    assert svc.sensor_out(svc.store.get_sensor("VOLT-001"))["status"] == "disabled"
    assert svc.pipeline.ingest(msg(battery_voltage=52), "http").rejected == 1


# --- alerts -----------------------------------------------------------------------------------------------
def test_threshold_alert_lifecycle():
    rule = Rule("bt", "battery_temperature", "above", warning=40, critical=50, hysteresis=2, for_s=60)
    events = []
    eng = AlertEngine(AlertConfig([rule]), None, "SITE", lambda k, a: events.append((a.severity, a.state)))
    ev = lambda v, s: eng.evaluate("T-1", "BAT-001", "battery", "battery_temperature", v,  # noqa: E731
                                   NOW + timedelta(seconds=s), "physical")
    assert ev(42, 0) == [] and ev(43, 30) == []  # must persist for 60 s
    opened = ev(44, 61)
    assert opened and opened[0].severity == "warning" and "Detected condition" in opened[0].message
    assert "faulty" not in opened[0].message.lower()
    assert ev(55, 90)[0].severity == "critical"  # escalation
    assert ev(39, 100) == []  # inside hysteresis band: stays active
    assert eng.active()[0].peak_value == 55
    cleared = ev(37.5, 110)
    assert cleared[0].state == "cleared" and eng.active() == []
    assert events[-1] == ("critical", "cleared")
    assert ev(41, 120) == []  # a new episode needs for_s again


def test_rule_scoping_and_config_validation(tmp_path):
    eng = AlertEngine(AlertConfig([Rule("rru", "temperature", "above", warning=70, categories=("radio_unit",))]),
                      None, "SITE")
    assert eng.evaluate("R-1", None, "router", "temperature", 90, NOW, "physical") == []
    assert eng.evaluate("R-2", None, "radio_unit", "temperature", 90, NOW, "physical")
    bad = tmp_path / "bad.yaml"
    bad.write_text("rules:\n  - {id: x, metric: humidity, above: {warning: 90, critical: 80}}\n")
    with pytest.raises(ValueError, match="critical level"):
        load_rules(bad)
    from app.core.config import BACKEND_DIR

    cfg = load_rules(BACKEND_DIR / "configs" / "telemetry" / "thresholds.yaml")
    assert len(cfg.rules) > 15 and cfg.communication_enabled


def test_alerts_and_cache_survive_restart(tmp_path):
    s1 = TelemetryService(tsettings(tmp_path))
    s1.start()
    add_battery(s1)
    s1.pipeline.ingest(msg(ts=NOW, battery_voltage=44.0), "http")  # below the critical 45 V level (for_s 30)
    s1.pipeline.ingest(msg(ts=NOW + timedelta(seconds=40), battery_voltage=44.1), "http")
    assert [a.rule_id for a in s1.alerts.active()] == ["battery_voltage_low"]
    s1.stop()
    s2 = TelemetryService(tsettings(tmp_path))
    s2.start()
    try:
        assert [a.rule_id for a in s2.alerts.active()] == ["battery_voltage_low"]
        assert s2.latest()[0]["value"] == 44.1
        assert s2.store.get_sensor("VOLT-001").last_seen_at is not None  # flushed at shutdown
        r = s2.pipeline.ingest(msg(ts=NOW + timedelta(seconds=10), battery_voltage=50), "http")
        assert r.out_of_order == 1  # ordering survives the restart
    finally:
        s2.stop()


# --- history ----------------------------------------------------------------------------------------------
def test_history_bucketing_and_empty_series(svc):
    add_battery(svc)
    msgs = [msg(ts=NOW + timedelta(seconds=10 * i), battery_voltage=50 + (i % 10) / 10) for i in range(360)]
    r = svc.pipeline.ingest_many(msgs, "http")
    assert r.stored == 360
    series = svc.history("battery_voltage", NOW, NOW + timedelta(hours=1), max_points=60)
    assert len(series) == 1 and len(series[0]["points"]) <= 60 and series[0]["bucket_s"] == 60
    p = series[0]["points"][0]
    assert p["min"] <= p["avg"] <= p["max"] and p["n"] == 6
    assert svc.history("humidity", NOW, NOW + timedelta(hours=1)) == []  # no data -> no chart
    bucket, pts = bucket_series([], NOW, NOW + timedelta(hours=1), 10)
    assert pts == [] and bucket == 360


# --- MQTT ---------------------------------------------------------------------------------------------------
class FakeMqttClient:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a, k))


def test_mqtt_adapter_with_mocked_client(svc):
    add_battery(svc, protocol="mqtt")
    s = svc.settings.model_copy(update={"mqtt_username": "dt", "mqtt_password": SecretStr("hunter2")})
    fake = FakeMqttClient()
    a = MqttAdapter(s, svc.pipeline, client_factory=lambda: fake)
    a.start()
    names = [c[0] for c in fake.calls]
    assert ["username_pw_set", "reconnect_delay_set", "connect_async", "loop_start"] == \
        [n for n in names if n in ("username_pw_set", "reconnect_delay_set", "connect_async", "loop_start")]
    a.on_connect(fake, None, None, 0)
    assert ("subscribe", ("bts/+/+/telemetry",), {"qos": 1}) in fake.calls and a.connected
    topic = f"bts/{svc.site_id}/VOLT-001/telemetry"
    payload = {"message_id": "a1", "timestamp": NOW.isoformat(), "readings": [{"metric": "battery_voltage",
                                                                              "value": 51.9}]}
    a.on_message(fake, None, SimpleNamespace(topic=topic, payload=json.dumps(payload).encode()))
    assert svc.latest()[0]["value"] == 51.9
    a.on_message(fake, None, SimpleNamespace(topic=topic, payload=b"{not json"))
    a.on_message(fake, None, SimpleNamespace(topic="bts/OTHER/VOLT-001/telemetry", payload=b"{}"))
    a.on_message(fake, None, SimpleNamespace(topic=topic, payload=json.dumps({**payload, "sensor_id": "X-9"}).encode()))
    a.on_message(fake, None, SimpleNamespace(topic="weird/topic", payload=b"{}"))
    st = a.status()
    assert (st["forwarded"], st["malformed"], st["wrong_site"], st["topic_mismatch"]) == (1, 2, 1, 1)
    assert "hunter2" not in json.dumps(st) and st["authenticated"] is True
    a.on_disconnect(fake, None, None, 7)
    assert not a.connected and "disconnected" in a.status()["last_error"]
    assert parse_topic("bts/S/X/telemetry") == ("S", "X") and parse_topic("bts/S/X") is None


# --- simulator --------------------------------------------------------------------------------------------
def test_simulated_site_model_is_bounded_and_physical():
    from app.core.config import BACKEND_DIR

    cfg = load_config(BACKEND_DIR / "configs" / "telemetry" / "simulator.yaml")
    m = SiteModel(cfg, random.Random(3), NOW)
    t = NOW
    socs, cabinet, discharge_seen = [], [], False
    for _ in range(2 * 24 * 60):  # two simulated days at 1-minute steps
        v = m.step(t, 60)
        socs.append(v["state_of_charge"])
        cabinet.append(v["internal_temperature"])
        assert 15 <= v["humidity"] <= 98 and 40 <= v["battery_voltage"] <= 54
        if not v["mains_available"]:
            discharge_seen = True
            assert v["battery_current"] < 0 and v["backup_power_active"] == 1 and v["psu"]["input_voltage"] == 0
        t += timedelta(minutes=1)
    assert 0 <= min(socs) and max(socs) <= 100 and 15 < min(cabinet) and max(cabinet) < 50
    assert discharge_seen  # 1.5 outages/day on average over two days (seeded)


def test_outage_scenario_discharges_and_depletes_battery():
    from app.core.config import BACKEND_DIR

    cfg = load_config(BACKEND_DIR / "configs" / "telemetry" / "simulator.yaml")
    cfg["events"] = {}
    m = SiteModel(cfg, random.Random(1), NOW)
    m.step(NOW, 1)
    m.trigger("mains_outage", 48 * 3600)
    t, v = NOW, None
    for _ in range(24 * 60):
        t += timedelta(minutes=1)
        v = m.step(t, 60)
    assert v["state_of_charge"] < 5 and v["rru"][1]["operational_status"] == 0  # radios shed load when exhausted


def test_simulator_lifecycle_dropout_and_isolation(svc):
    sim = svc.simulator
    sim.tick_s = 0.05
    sim.start(interval_scale=50, seed=7)
    try:
        assert sim.running
        with pytest.raises(SimulatorError, match="already running"):
            sim.start()
        time.sleep(1.0)
        out = sim.scenario("sensor_dropout", 60, "SIM-SITE-ENV")
        assert out["sensor_id"] == "SIM-SITE-ENV"
        n_before = svc.store.measurements_page(1, 0, sensor_id="SIM-SITE-ENV")[0]
        time.sleep(0.6)
        assert svc.store.measurements_page(1, 0, sensor_id="SIM-SITE-ENV")[0] == n_before  # silent
        assert svc.store.measurements_page(1, 0, sensor_id="SIM-BAT-001-VOLT")[0] > 0
        with pytest.raises(SimulatorError):
            sim.scenario("sensor_dropout", 10, "VOLT-001")
    finally:
        sim.stop()
    assert not sim.running
    sensors = [s for s in svc.store.list_sensors()]
    assert sensors and all(s.data_source == "simulated" and s.protocol == "simulator" and s.sensor_id.startswith("SIM-")
                           for s in sensors)
    _, rows = svc.store.measurements_page(1000, 0)
    assert rows and all(r.data_source == "simulated" for r in rows)
    # the stopped simulator's sensors go offline, but that is not reported as a communication failure
    svc.check_status(datetime.now(UTC) + timedelta(hours=1))
    assert not [a for a in svc.alerts.active() if a.rule_id == "sensor_communication"]


def test_simulator_refuses_production(tmp_path):
    sim = Simulator(tmp_path / "x.yaml", None, None, app_env="production")
    with pytest.raises(SimulatorError, match="production"):
        sim.start()


# --- API + WebSocket ---------------------------------------------------------------------------------------
@pytest.fixture
def client(tmp_path):
    app = create_app(tsettings(tmp_path, telemetry_ingest_api_key=SecretStr("k3y")))
    with TestClient(app) as c:
        yield c


def _register(client):
    assert client.post("/api/telemetry/equipment", json={"equipment_id": "BAT-001", "category": "battery",
                                                         "name": "Battery string 1"}).status_code == 201
    r = client.post("/api/telemetry/sensors", json={"sensor_id": "VOLT-001", "name": "Battery voltage",
                                                    "sensor_type": "voltage", "protocol": "http",
                                                    "metrics": ["battery_voltage"], "equipment_id": "BAT-001"})
    assert r.status_code == 201 and r.json()["status"] == "unknown"


def test_api_registration_ingest_auth_and_queries(client):
    _register(client)
    assert client.post("/api/telemetry/sensors", json={"sensor_id": "SIM-X", "name": "x", "sensor_type": "voltage",
                                                       "protocol": "simulator", "metrics": ["humidity"],
                                                       "category": "site"}).status_code == 422
    assert client.post("/api/telemetry/equipment", json={"equipment_id": "BAT-001", "category": "battery",
                                                         "name": "dup"}).status_code == 409
    body = {"sensor_id": "VOLT-001", "readings": [{"metric": "battery_voltage", "value": 52.3, "unit": "V"}]}
    assert client.post("/api/telemetry/ingest", json=body).status_code == 401
    assert client.post("/api/telemetry/ingest", json=body, headers={"X-API-Key": "nope"}).status_code == 401
    r = client.post("/api/telemetry/ingest", json=body, headers={"X-API-Key": "k3y"})
    assert r.status_code == 200 and r.json()["stored"] == 1
    assert client.post("/api/telemetry/ingest", json={"sensor_id": "VOLT-001", "readings": []},
                       headers={"X-API-Key": "k3y"}).status_code == 422
    latest = client.get("/api/telemetry/latest", params={"equipment_id": "BAT-001"}).json()
    assert latest[0]["value"] == 52.3 and latest[0]["data_source"] == "physical" and latest[0]["current"]
    eq = client.get("/api/telemetry/equipment/BAT-001/telemetry").json()
    assert eq["equipment"]["sensor_ids"] == ["VOLT-001"] and eq["sensors"][0]["status"] == "online"
    page = client.get("/api/telemetry/sensors", params={"limit": 1, "category": "battery"}).json()
    assert page["total"] == 1 and len(page["items"]) == 1
    assert client.get("/api/telemetry/history", params={"metric": "nope"}).status_code == 422
    assert client.get("/api/telemetry/history", params={"metric": "battery_voltage", "range": "2y"}).status_code == 422
    assert client.get("/api/telemetry/history", params={"metric": "battery_voltage"}).json()["series"]
    r = client.put("/api/telemetry/sensors/VOLT-001/equipment", json={"equipment_id": None})
    assert r.status_code == 200 and r.json()["equipment_id"] is None
    assert client.get("/api/telemetry/sensors/NOPE").status_code == 404
    st = client.get("/api/telemetry/status").json()
    assert st["available"] and st["ingest_api_key_configured"] and "k3y" not in json.dumps(st)
    assert client.get("/api/health").json()["status"] == "ok"  # vision API unaffected


def test_api_alert_ack_flow(client):
    _register(client)
    h = {"X-API-Key": "k3y"}
    t = datetime.now(UTC) - timedelta(minutes=2)
    for i, v in enumerate((44.0, 44.1)):
        client.post("/api/telemetry/ingest", headers=h, json={
            "sensor_id": "VOLT-001", "timestamp": (t + timedelta(seconds=40 * i)).isoformat(),
            "readings": [{"metric": "battery_voltage", "value": v}]})
    alerts = client.get("/api/telemetry/alerts", params={"state": "active"}).json()
    assert alerts["total"] == 1 and alerts["items"][0]["severity"] == "critical"
    aid = alerts["items"][0]["id"]
    r = client.post(f"/api/telemetry/alerts/{aid}/ack", json={"acknowledged_by": "harsha", "note": "checking"})
    assert r.status_code == 200 and r.json()["acknowledged"] and r.json()["state"] == "active"
    assert client.post("/api/telemetry/alerts/999/ack", json={"acknowledged_by": "x"}).status_code == 404


def test_websocket_snapshot_and_live_updates(client):
    _register(client)
    with client.websocket_connect("/ws/telemetry") as ws:
        snap = ws.receive_json()
        assert snap["type"] == "snapshot" and snap["sensors"][0]["sensor_id"] == "VOLT-001"
        client.post("/api/telemetry/ingest", headers={"X-API-Key": "k3y"},
                    json={"sensor_id": "VOLT-001", "readings": [{"metric": "battery_voltage", "value": 51.0}]})
        batch = ws.receive_json()
        assert batch["type"] == "batch"
        items = [i for e in batch["events"] if e["type"] == "measurements" for i in e["items"]]
        assert items[0]["value"] == 51.0 and items[0]["sensor_id"] == "VOLT-001"


def test_simulator_api_and_ingest_closed_without_key(tmp_path):
    with TestClient(create_app(tsettings(tmp_path))) as c:
        r = c.post("/api/telemetry/ingest", json={"sensor_id": "A-1", "readings": [{"metric": "humidity",
                                                                                    "value": 1}]})
        assert r.status_code == 403
        assert c.post("/api/telemetry/simulator/scenario", json={"scenario": "door_open"}).status_code == 409
        assert c.post("/api/telemetry/simulator/start", json={"interval_scale": 20, "seed": 1}).json()["running"]
        time.sleep(1.5)
        assert c.get("/api/telemetry/summary").json()["sensors_simulated"] == 12
        assert c.post("/api/telemetry/simulator/stop").json()["running"] is False


def test_unreachable_database_disables_only_telemetry(tmp_path):
    s = tsettings(tmp_path, telemetry_database_url=SecretStr("postgresql+psycopg://u:secretpw@127.0.0.1:1/none"))
    with TestClient(create_app(s)) as c:
        r = c.get("/api/telemetry/summary")
        assert r.status_code == 503 and "secretpw" not in r.text
        assert c.get("/api/telemetry/status").json()["available"] is False
        assert c.get("/api/health").json()["model_loaded"] is True
