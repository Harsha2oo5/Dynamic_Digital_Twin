"""Simulated BTS site: a physical model of the site plus an emitter that sends its readings through the normal
ingestion pipeline (protocol "simulator", data_source "simulated").

Model (per step of dt seconds of site time):
  ambient     daily sine cycle (+ heat wave) with bounded Ornstein-Uhlenbeck noise; humidity anti-correlated
  traffic     daily profile, peak around 20:00 local time
  load        radio units 180 W + 320 W x traffic, router 70 W + 25 W x traffic, 40 W auxiliary
  mains       random outages; while mains is lost the battery carries the whole DC load
  battery     24-cell lead-acid string: SOC integrates the current (charge with taper up to max_charge_a while mains
              is present, discharge = load / voltage during an outage); voltage = float voltage on mains, open-circuit
              voltage(SOC) minus I*R during discharge; temperature lags the cabinet plus resistive heating
  cabinet     first-order lag towards ambient + equipment heat; much hotter when the fan fails; door events
  radio units shut down (operational_status 0) when the battery is exhausted
Every sensor adds small measurement noise. Sensors can drop out (no messages) and recover.
"""

import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from app.telemetry.schemas import IngestMessage, IngestReading

logger = logging.getLogger(__name__)

# injected equipment faults for anomaly-detection validation (in addition to mains_outage, door_open, fan_failure,
# heat_wave and sensor_dropout); "sudden" ones act at full strength at once, "gradual" ones ramp over their duration
FAULTS = {"temperature_spike": "sudden", "power_anomaly": "sudden", "voltage_fluctuation": "sudden",
          "battery_degradation": "gradual", "psu_overheating": "gradual"}

CELLS = 24
FLOAT_V = 53.5
R_INTERNAL = 0.03
SIMULATOR_PROTOCOL = "simulator"


class SimulatorError(RuntimeError):
    pass


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    ids = [e["id"] for e in cfg.get("equipment", [])] + [s["id"] for e in cfg.get("equipment", [])
                                                          for s in e.get("sensors", [])]
    ids += [s["id"] for s in cfg.get("site_sensors", [])]
    if len(ids) != len(set(ids)):
        raise SimulatorError("simulator config has duplicate IDs")
    bad = [i for i in ids if not i.startswith("SIM-")]
    if bad:
        raise SimulatorError(f"simulated device IDs must start with SIM- (got {bad})")
    return cfg


@dataclass
class SiteModel:
    cfg: dict
    rng: random.Random
    now: datetime
    ambient_noise: float = 0.0
    humidity_noise: float = 0.0
    traffic_noise: float = 0.0
    cabinet_c: float | None = None
    battery_c: float | None = None
    soc: float = 96.0
    soh: float = 94.0
    current: float = 0.0
    voltage: float = FLOAT_V
    mains: bool = True
    outage_until: datetime | None = None
    door_until: datetime | None = None
    fan_fail_until: datetime | None = None
    heat_wave_until: datetime | None = None
    faults: dict = field(default_factory=dict)  # injected equipment faults: name -> (start, until)
    values: dict = field(default_factory=dict)

    def __post_init__(self):
        b = self.cfg.get("battery", {})
        self.soc = float(b.get("initial_soc", 96.0))
        self.soh = float(b.get("state_of_health", 94.0))

    # --- events ---------------------------------------------------------------------------------
    def _chance(self, per_day: float, dt: float) -> bool:
        return per_day > 0 and self.rng.random() < per_day * dt / 86400.0

    def _minutes(self, key: str, default: tuple[float, float]) -> timedelta:
        lo, hi = self.cfg.get("events", {}).get(key, default)
        return timedelta(minutes=self.rng.uniform(lo, hi))

    def trigger(self, scenario: str, duration_s: float) -> None:
        until = self.now + timedelta(seconds=duration_s)
        if scenario == "mains_outage":
            self.outage_until = until
        elif scenario == "door_open":
            self.door_until = until
        elif scenario == "fan_failure":
            self.fan_fail_until = until
        elif scenario == "heat_wave":
            self.heat_wave_until = until
        elif scenario in FAULTS:
            self.faults[scenario] = (self.now, until)
        else:
            raise SimulatorError(f"unknown scenario {scenario}")

    @staticmethod
    def _active(until: datetime | None, now: datetime) -> bool:
        return until is not None and now < until

    # --- physics --------------------------------------------------------------------------------
    def step(self, now: datetime, dt: float) -> dict:
        self.now = now
        ev = self.cfg.get("events", {})
        if not self._active(self.outage_until, now) and self._chance(ev.get("mains_outages_per_day", 0), dt):
            self.outage_until = now + self._minutes("mains_outage_minutes", (10, 45))
        if not self._active(self.door_until, now) and self._chance(ev.get("door_openings_per_day", 0), dt):
            self.door_until = now + self._minutes("door_open_minutes", (3, 15))
        self.mains = not self._active(self.outage_until, now)
        fan_on = not self._active(self.fan_fail_until, now)
        door = self._active(self.door_until, now)
        heat = 8.0 if self._active(self.heat_wave_until, now) else 0.0

        a = self.cfg.get("ambient", {})
        hour = ((now + timedelta(hours=self.cfg.get("utc_offset_h", 0))).hour
                + (now.minute + now.second / 60) / 60)
        k = math.exp(-dt / 600.0)  # OU noise, 10-minute correlation
        self.ambient_noise = self.ambient_noise * k + self.rng.gauss(0, 0.6) * math.sqrt(1 - k * k)
        self.humidity_noise = self.humidity_noise * k + self.rng.gauss(0, 3.0) * math.sqrt(1 - k * k)
        self.traffic_noise = self.traffic_noise * k + self.rng.gauss(0, 0.05) * math.sqrt(1 - k * k)
        daily = math.sin(2 * math.pi * (hour - 9) / 24)
        ambient = a.get("mean_c", 27) + a.get("daily_amplitude_c", 6) * daily + heat + self.ambient_noise
        humidity = min(98.0, max(15.0, a.get("humidity_mean", 65) - a.get("humidity_amplitude", 15) * daily
                                 + self.humidity_noise))
        traffic = min(1.0, max(0.05, 0.55 + 0.45 * math.cos(2 * math.pi * (hour - 20) / 24) + self.traffic_noise))

        exhausted = not self.mains and self.soc <= 3.0
        rru = {} if exhausted else {i: 180 + 320 * traffic for i in (1, 2)}
        router = 0.0 if exhausted else 70 + 25 * traffic
        load_w = sum(rru.values()) + router + 40.0

        b = self.cfg.get("battery", {})
        cap = float(b.get("capacity_ah", 200.0))
        max_i = float(b.get("max_charge_a", 20.0))
        ocv = CELLS * (1.95 + 0.18 * self.soc / 100.0)
        if self.mains:
            self.current = max_i * min(1.0, (100.0 - self.soc) / 10.0) if self.soc < 99.5 else 0.3
            self.voltage = FLOAT_V - max(0.0, 90.0 - self.soc) * 0.03
        else:
            self.current = -load_w / max(ocv, 40.0) if load_w > 40.0 else -1.0
            self.voltage = ocv + self.current * R_INTERNAL
        self.soc = min(100.0, max(0.0, self.soc + self.current * dt / 3600.0 / cap * 100.0))

        lag = lambda cur, target, tau: target if cur is None else cur + (target - cur) * min(1.0, dt / tau)  # noqa: E731
        # heat-exchanger cabinet: a few degrees above ambient with the fan, much hotter without it
        cab_target = ambient + (3.0 if fan_on else 14.0) + load_w / 500.0 + (1.5 if door else 0.0)
        self.cabinet_c = lag(self.cabinet_c, cab_target, 600.0)
        self.battery_c = lag(self.battery_c, self.cabinet_c + abs(self.current) * 0.05, 1800.0)
        dc_bus = self.voltage
        psu_out_a = (load_w + max(self.current, 0) * dc_bus) / dc_bus if self.mains else 0.0

        self.values = {
            "ambient_temperature": ambient, "ambient_humidity": humidity,
            "mains_available": int(self.mains), "backup_power_active": int(not self.mains),
            "site_power": load_w,
            "internal_temperature": self.cabinet_c, "humidity": min(98.0, max(10.0, humidity - 12 + (8 if door else 0))),
            "door_open": int(door), "smoke_detected": 0, "fan_status": int(fan_on),
            "psu": {"input_voltage": 230.0 if self.mains else 0.0, "output_voltage": dc_bus if self.mains else 0.0,
                    "output_current": psu_out_a, "power_consumption": psu_out_a * dc_bus / 0.94 if self.mains else 0.0,
                    "temperature": self.cabinet_c + 6 + 8 * load_w / 1200.0, "operational_status": 1},
            "battery_voltage": self.voltage, "battery_current": self.current,
            "battery_state": 1 if self.current > 0.5 else (-1 if self.current < -0.5 else 0),
            "battery_temperature": self.battery_c, "state_of_charge": self.soc, "state_of_health": self.soh,
            "router": {"temperature": self.cabinet_c + 15 + 5 * traffic, "power_consumption": router,
                       "supply_voltage": dc_bus, "operational_status": int(router > 0),
                       "network_available": int(router > 0)},
            "rru": {i: {"temperature": ambient + (12 + 15 * traffic if i in rru else 3), "power_consumption": rru.get(i, 0.0),
                        "supply_voltage": dc_bus, "operational_status": int(i in rru), "network_available": int(i in rru),
                        "vswr": 1.12 + 0.04 * i} for i in (1, 2)},
        }
        self._apply_faults(now)
        return self.values

    def fault_progress(self, name: str, now: datetime) -> float | None:
        """0..1 progress of an injected fault, None when it is not active."""
        f = self.faults.get(name)
        if f is None or not (f[0] <= now < f[1]):
            return None
        return min(1.0, (now - f[0]).total_seconds() / max((f[1] - f[0]).total_seconds(), 1.0))

    def _apply_faults(self, now: datetime) -> None:
        """Injected equipment faults (validation scenarios). They change what the equipment does, so sensors report
        the faulty behaviour; they end (recovery) when their duration is over."""
        v = self.values
        if self.fault_progress("temperature_spike", now) is not None:  # sudden: radio unit 1 cooling problem
            v["rru"][1]["temperature"] += 18.0
        if self.fault_progress("power_anomaly", now) is not None and v["rru"][2]["operational_status"]:
            v["rru"][2]["power_consumption"] += 250.0  # radio unit 2 draws far more than its traffic explains
        if self.fault_progress("voltage_fluctuation", now) is not None and self.mains:
            sign = 1 if int(now.timestamp() // 60) % 2 else -1  # rectifier output oscillating minute to minute
            v["psu"]["output_voltage"] += sign * self.rng.uniform(0.8, 1.4)
        p = self.fault_progress("battery_degradation", now)
        if p is not None:  # gradual: voltage sags, string runs warmer, health declines over the fault duration
            v["battery_voltage"] -= 2.5 * p
            v["battery_temperature"] += 6.0 * p
            v["state_of_health"] -= 8.0 * p
        p = self.fault_progress("psu_overheating", now)
        if p is not None:  # gradual: rectifier fans clogging
            v["psu"]["temperature"] += 15.0 * p


# measurement noise (standard deviation, canonical units) per metric
NOISE = {"battery_voltage": 0.03, "battery_current": 0.1, "battery_temperature": 0.2, "input_voltage": 1.5,
         "output_voltage": 0.03, "output_current": 0.2, "power_consumption": 3.0, "temperature": 0.3,
         "internal_temperature": 0.2, "humidity": 0.8, "supply_voltage": 0.03, "ambient_temperature": 0.2,
         "ambient_humidity": 0.8, "vswr": 0.01, "state_of_charge": 0.0, "state_of_health": 0.0}
BOUNDS = {"humidity": (0, 100), "ambient_humidity": (0, 100), "state_of_charge": (0, 100), "vswr": (1.0, 20),
          "power_consumption": (0, None), "output_current": (0, None), "input_voltage": (0, None)}


def read_metric(values: dict, equipment: dict | None, metric: str) -> float | None:
    """The model value a sensor on `equipment` (None = site) reports for `metric`."""
    cat = equipment["category"] if equipment else "site"
    if cat == "power_supply_unit":
        return values["psu"].get(metric)
    if cat == "router":
        return values["router"].get(metric)
    if cat == "radio_unit":
        idx = int(equipment["id"].rsplit("-", 1)[-1])
        return values["rru"].get(1 + (idx - 1) % 2, {}).get(metric)
    if cat == "site" and metric == "power_consumption":
        return values["site_power"]
    return values.get(metric)


class Simulator:
    """Runs the model in a background thread and emits readings through `pipeline.ingest(..., "simulator")`."""

    def __init__(self, config_path: Path, pipeline, register, app_env: str = "development", tick_s: float = 1.0):
        self.config_path, self.pipeline, self.register = Path(config_path), pipeline, register
        self.app_env, self.tick_s = app_env, tick_s
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.model: SiteModel | None = None
        self.cfg: dict | None = None
        self.sensors: list[tuple[dict, dict | None]] = []  # (sensor cfg, equipment cfg or None)
        self.dropouts: dict[str, datetime] = {}
        self.interval_scale = 1.0
        self.started_at: datetime | None = None
        self.messages = 0
        self.backfill: dict | None = None
        self.backfilling = False
        self.last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        m = self.model
        return {"running": self.running, "backfilling": self.backfilling, "started_at": self.started_at, "interval_scale": self.interval_scale,
                "messages_sent": self.messages, "sensors": len(self.sensors), "backfill": self.backfill,
                "dropouts": {k: v for k, v in self.dropouts.items() if m is None or v > m.now},
                "events": None if m is None else {
                    "mains_outage_until": m.outage_until if m._active(m.outage_until, m.now) else None,
                    "door_open_until": m.door_until if m._active(m.door_until, m.now) else None,
                    "fan_failure_until": m.fan_fail_until if m._active(m.fan_fail_until, m.now) else None,
                    "heat_wave_until": m.heat_wave_until if m._active(m.heat_wave_until, m.now) else None},
                "last_error": self.last_error, "data_source": "simulated"}

    def _prepare(self, seed: int | None) -> None:
        self.cfg = load_config(self.config_path)
        self.sensors = [(s, e) for e in self.cfg.get("equipment", []) for s in e.get("sensors", [])]
        self.sensors += [(s, None) for s in self.cfg.get("site_sensors", [])]
        self.register(self.cfg)
        self.model = SiteModel(self.cfg, random.Random(seed), datetime.now(UTC))

    def _message(self, sensor: dict, equipment: dict | None, at: datetime, seq: int) -> IngestMessage:
        rng = self.model.rng
        readings = []
        for metric in sensor["metrics"]:
            v = read_metric(self.model.values, equipment, metric)
            if v is None:
                readings.append(IngestReading(metric=metric, value=None))
                continue
            v = float(v) + rng.gauss(0, NOISE.get(metric, 0.0))
            lo, hi = BOUNDS.get(metric, (None, None))
            v = max(lo, v) if lo is not None else v
            v = min(hi, v) if hi is not None else v
            readings.append(IngestReading(metric=metric, value=round(v, 3)))
        return IngestMessage(sensor_id=sensor["id"], message_id=f"{sensor['id']}-{int(at.timestamp() * 1000)}-{seq}",
                             timestamp=at, readings=readings)

    def _emit(self, at: datetime, due: dict[str, datetime], live: bool) -> None:
        for sensor, equipment in self.sensors:
            sid = sensor["id"]
            if due[sid] > at:
                continue
            due[sid] = at + timedelta(seconds=sensor["interval_s"] / self.interval_scale)
            until = self.dropouts.get(sid)
            if until is not None and at < until:
                continue  # sensor is silent
            ev = self.cfg.get("events", {})
            if live and self.rng_chance(ev.get("sensor_dropouts_per_hour", 0) * 24,
                                        sensor["interval_s"] / self.interval_scale):
                lo, hi = ev.get("sensor_dropout_seconds", (45, 600))
                self.dropouts[sid] = at + timedelta(seconds=self.model.rng.uniform(lo, hi))
                logger.info("Simulator: sensor %s drops out until %s", sid, self.dropouts[sid])
                continue
            self.pipeline.ingest(self._message(sensor, equipment, at, self.messages), SIMULATOR_PROTOCOL,
                                 received_at=at if not live else None)
            self.messages += 1

    def rng_chance(self, per_day: float, dt: float) -> bool:
        return self.model._chance(per_day, dt)

    def _backfill(self, hours: float) -> None:
        """Clearly-simulated history: the model runs from `hours` ago to now at a coarse step, without dropouts."""
        now = datetime.now(UTC)
        n_sensors = max(1, len(self.sensors))
        step = max(30.0, hours * 3600 * n_sensors / 60_000)  # cap at roughly 60k messages
        t = now - timedelta(hours=hours)
        self.model.now = t
        saved_scale, self.interval_scale = self.interval_scale, 1.0
        due = {s["id"]: t for s, _ in self.sensors}
        sent0 = self.messages
        self.backfilling = True
        try:
            while t < now and not self._stop.is_set():
                batch = []
                for _ in range(50):  # ~50 model steps per database transaction
                    if t >= now:
                        break
                    self.model.step(t, step)
                    for sensor, equipment in self.sensors:
                        if due[sensor["id"]] <= t:
                            due[sensor["id"]] = t + timedelta(seconds=max(step, sensor["interval_s"]))
                            batch.append(self._message(sensor, equipment, t, self.messages))
                            self.messages += 1
                    t += timedelta(seconds=step)
                self.pipeline.ingest_many(batch, SIMULATOR_PROTOCOL, received_at=t)
        finally:
            self.backfilling = False
        self.interval_scale = saved_scale
        self.backfill = {"hours": hours, "step_s": round(step, 1), "messages": self.messages - sent0}

    def start(self, interval_scale: float = 1.0, backfill_hours: float = 0.0, seed: int | None = None) -> dict:
        if self.app_env == "production":
            raise SimulatorError("the simulator is disabled when APP_ENV=production")
        with self._lock:
            if self.running:
                raise SimulatorError("simulator is already running")
            self._stop.clear()
            self.interval_scale, self.messages, self.backfill, self.last_error = interval_scale, 0, None, None
            self.dropouts = {}
            self._prepare(seed)
            self.started_at = datetime.now(UTC)
            self._thread = threading.Thread(target=self._run, args=(backfill_hours,), name="telemetry-simulator",
                                            daemon=True)
            self._thread.start()
        logger.info("Simulator started (%d sensors, interval scale %.1f, backfill %.1f h)", len(self.sensors),
                    interval_scale, backfill_hours)
        return self.status()

    def _run(self, backfill_hours: float) -> None:
        try:
            if backfill_hours > 0:
                self._backfill(backfill_hours)
            now = datetime.now(UTC)
            self.model.now = now
            due = {s["id"]: now for s, _ in self.sensors}
            last = time.monotonic()
            while not self._stop.is_set():
                mono = time.monotonic()
                dt, last = mono - last, mono
                now = datetime.now(UTC)
                self.model.step(now, max(dt, 1e-3) * self.interval_scale)
                self._emit(now, due, live=True)
                self._stop.wait(self.tick_s)
        except Exception as exc:  # noqa: BLE001 - report through status instead of dying silently
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Simulator stopped by an error")

    def stop(self, timeout: float = 10.0) -> dict:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout)
            if t.is_alive():
                logger.error("Simulator thread did not stop within %.0f s", timeout)
        self._thread = None
        logger.info("Simulator stopped after %d messages", self.messages)
        return self.status()

    def scenario(self, name: str, duration_s: float, sensor_id: str | None = None) -> dict:
        if not self.running or self.model is None:
            raise SimulatorError("start the simulator first")
        if self.backfilling:
            raise SimulatorError("history backfill is still running; try again when it has finished")
        if name == "sensor_dropout":
            ids = [s["id"] for s, _ in self.sensors]
            if sensor_id is not None and sensor_id not in ids:
                raise SimulatorError(f"{sensor_id} is not a simulated sensor")
            sid = sensor_id or self.model.rng.choice(ids)
            self.dropouts[sid] = datetime.now(UTC) + timedelta(seconds=duration_s)
            return {"scenario": name, "sensor_id": sid, "until": self.dropouts[sid]}
        with self._lock:
            self.model.trigger(name, duration_s)
        return {"scenario": name, "until": datetime.now(UTC) + timedelta(seconds=duration_s)}
