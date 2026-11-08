"""Metric catalogue: names, canonical units, accepted unit conversions, plausibility ranges and state encodings.

A reading is normalised to its metric's canonical unit. Values outside the *plausibility* range are physically
implausible for a BTS (a sensor fault or a unit error), so they are stored with quality "invalid" and never shown
as a measurement. Plausibility is not an alarm threshold; alarm thresholds live in configs/telemetry/thresholds.yaml.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Metric:
    name: str
    unit: str  # canonical unit
    label: str
    kind: str = "gauge"  # gauge | state
    min: float | None = None  # plausibility range (inclusive)
    max: float | None = None
    categories: tuple[str, ...] = ()  # equipment categories this metric usually belongs to ("site" = site level)
    states: dict[int, str] = field(default_factory=dict)  # for kind == "state": encoded value -> meaning


# unit alias -> (canonical unit, factor, offset): canonical = value * factor + offset
UNIT_CONVERSIONS: dict[str, tuple[str, float, float]] = {
    "v": ("V", 1.0, 0.0), "volt": ("V", 1.0, 0.0), "volts": ("V", 1.0, 0.0), "mv": ("V", 0.001, 0.0),
    "kv": ("V", 1000.0, 0.0),
    "a": ("A", 1.0, 0.0), "amp": ("A", 1.0, 0.0), "amps": ("A", 1.0, 0.0), "ma": ("A", 0.001, 0.0),
    "w": ("W", 1.0, 0.0), "watt": ("W", 1.0, 0.0), "watts": ("W", 1.0, 0.0), "kw": ("W", 1000.0, 0.0),
    "c": ("°C", 1.0, 0.0), "°c": ("°C", 1.0, 0.0), "degc": ("°C", 1.0, 0.0), "celsius": ("°C", 1.0, 0.0),
    "f": ("°C", 5 / 9, -32 * 5 / 9), "°f": ("°C", 5 / 9, -32 * 5 / 9), "degf": ("°C", 5 / 9, -32 * 5 / 9),
    "k": ("°C", 1.0, -273.15),
    "%": ("%", 1.0, 0.0), "percent": ("%", 1.0, 0.0), "pct": ("%", 1.0, 0.0), "%rh": ("%", 1.0, 0.0),
    "ratio": ("%", 100.0, 0.0),  # 0..1 -> %
    "dbm": ("dBm", 1.0, 0.0), "db": ("dB", 1.0, 0.0),
    "state": ("state", 1.0, 0.0), "bool": ("state", 1.0, 0.0), "": ("state", 1.0, 0.0),
}

ON_OFF = {0: "off", 1: "on"}
UP_DOWN = {0: "down", 1: "up"}

METRICS: dict[str, Metric] = {m.name: m for m in [
    # battery
    Metric("battery_voltage", "V", "Battery voltage", min=0, max=70, categories=("battery",)),
    Metric("battery_current", "A", "Battery current (+ charging, − discharging)", min=-500, max=500,
           categories=("battery",)),
    Metric("battery_temperature", "°C", "Battery temperature", min=-40, max=100, categories=("battery",)),
    Metric("state_of_charge", "%", "State of charge", min=0, max=100, categories=("battery",)),
    Metric("state_of_health", "%", "State of health", min=0, max=100, categories=("battery",)),
    Metric("battery_state", "state", "Charging state", kind="state", min=-1, max=1, categories=("battery",),
           states={-1: "discharging", 0: "idle", 1: "charging"}),
    # power supply unit
    Metric("input_voltage", "V", "Input voltage (AC)", min=0, max=300, categories=("power_supply_unit",)),
    Metric("output_voltage", "V", "Output voltage (DC)", min=0, max=70, categories=("power_supply_unit",)),
    Metric("output_current", "A", "Output current", min=0, max=1000, categories=("power_supply_unit",)),
    # shared electrical / thermal
    Metric("power_consumption", "W", "Power consumption", min=0, max=50000,
           categories=("power_supply_unit", "router", "radio_unit", "site")),
    Metric("supply_voltage", "V", "Supply voltage (DC)", min=0, max=70, categories=("router", "radio_unit")),
    Metric("temperature", "°C", "Operating temperature", min=-40, max=125,
           categories=("power_supply_unit", "router", "radio_unit")),
    Metric("operational_status", "state", "Operational status", kind="state", min=0, max=1,
           categories=("power_supply_unit", "router", "radio_unit"), states=UP_DOWN),
    # cabinet
    Metric("internal_temperature", "°C", "Internal temperature", min=-40, max=100, categories=("equipment_cabinet",)),
    Metric("humidity", "%", "Relative humidity", min=0, max=100, categories=("equipment_cabinet",)),
    Metric("door_open", "state", "Door", kind="state", min=0, max=1, categories=("equipment_cabinet",),
           states={0: "closed", 1: "open"}),
    Metric("smoke_detected", "state", "Smoke detector", kind="state", min=0, max=1, categories=("equipment_cabinet",),
           states={0: "clear", 1: "smoke"}),
    Metric("fan_status", "state", "Cooling fan", kind="state", min=0, max=1, categories=("equipment_cabinet",),
           states=ON_OFF),
    # router / radio unit
    Metric("network_available", "state", "Network availability", kind="state", min=0, max=1,
           categories=("router", "radio_unit"), states=UP_DOWN),
    Metric("rssi", "dBm", "Received signal strength", min=-150, max=0, categories=("radio_unit",)),
    Metric("vswr", "ratio", "VSWR", min=1, max=20, categories=("radio_unit", "antenna", "cable")),
    # site
    Metric("ambient_temperature", "°C", "Ambient temperature", min=-50, max=70, categories=("site",)),
    Metric("ambient_humidity", "%", "Ambient humidity", min=0, max=100, categories=("site",)),
    Metric("mains_available", "state", "Mains power", kind="state", min=0, max=1, categories=("site",),
           states={0: "lost", 1: "available"}),
    Metric("backup_power_active", "state", "Backup power", kind="state", min=0, max=1, categories=("site",),
           states={0: "inactive", 1: "active"}),
]}

SENSOR_TYPES = ("temperature", "voltage", "current", "power", "humidity", "door", "smoke", "fan", "battery_monitor",
                "psu_monitor", "network", "rf", "environment", "multi")
PROTOCOLS = ("http", "mqtt", "simulator", "modbus")
EQUIPMENT_CATEGORIES = ("antenna", "router", "cable", "battery", "power_supply_unit", "equipment_cabinet",
                        "radio_unit", "site")


class UnitError(ValueError):
    pass


def normalise(metric: str, value: float, unit: str | None) -> tuple[float, str]:
    """Convert `value` from `unit` to the metric's canonical unit. Unknown or incompatible units raise UnitError."""
    m = METRICS.get(metric)
    if m is None:
        raise UnitError(f"unknown metric {metric!r}")
    if m.unit == "ratio":  # dimensionless metrics such as VSWR are not converted
        if unit not in (None, "", "ratio", ":1"):
            raise UnitError(f"{metric} is dimensionless, got unit {unit!r}")
        return float(value), "ratio"
    if unit is None:
        return float(value), m.unit  # caller sends canonical units
    conv = UNIT_CONVERSIONS.get(unit.strip().lower())
    if conv is None:
        if unit == m.unit:
            return float(value), m.unit
        raise UnitError(f"unknown unit {unit!r} for {metric}")
    canonical, factor, off = conv
    if canonical != m.unit:
        raise UnitError(f"unit {unit!r} is {canonical}, but {metric} is measured in {m.unit}")
    return float(value) * factor + off, m.unit


def plausible(metric: str, value: float) -> bool:
    m = METRICS[metric]
    if m.kind == "state" and m.states and int(value) != value:
        return False
    if m.kind == "state" and m.states and int(value) not in m.states:
        return False
    return (m.min is None or value >= m.min) and (m.max is None or value <= m.max)


def catalogue() -> list[dict]:
    return [{"name": m.name, "unit": m.unit, "label": m.label, "kind": m.kind, "min": m.min, "max": m.max,
             "categories": list(m.categories), "states": {str(k): v for k, v in m.states.items()}}
            for m in METRICS.values()]
