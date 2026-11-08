"""Analytics configuration: YAML -> validated dict, plus a stable hash recorded with every anomaly model."""

import hashlib
import json
from pathlib import Path

import yaml

from app.core.config import BACKEND_DIR
from app.telemetry.catalog import EQUIPMENT_CATEGORIES, METRICS

DEFAULT_FILE = BACKEND_DIR / "configs" / "analytics" / "anomaly.yaml"
UNIVARIATE = ("robust_z", "iqr", "rate_of_change", "forecast_residual", "transitions")
REQUIRED = ("grid_s", "baseline_buckets", "min_history_buckets", "short_buckets", "max_gap_buckets", "severity",
            "episode", "detectors", "scale_floor", "categories", "health", "maintenance")


class ConfigError(ValueError):
    pass


def parse(text: str) -> dict:
    try:
        cfg = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from None
    validate(cfg)
    return cfg


def load(path: Path | None = None) -> tuple[dict, str]:
    text = Path(path or DEFAULT_FILE).read_text(encoding="utf-8")
    return parse(text), text


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def validate(cfg: dict) -> None:
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ConfigError(f"missing keys: {missing}")
    for k in ("grid_s", "baseline_buckets", "min_history_buckets", "short_buckets"):
        if not isinstance(cfg[k], (int, float)) or cfg[k] <= 0:
            raise ConfigError(f"{k} must be a positive number")
    if cfg["min_history_buckets"] > cfg["baseline_buckets"]:
        raise ConfigError("min_history_buckets cannot exceed baseline_buckets")
    sev = cfg["severity"]
    if not 0 < sev.get("warning", 0) < sev.get("critical", 0):
        raise ConfigError("severity needs 0 < warning < critical")
    for name, metric_floor in cfg["scale_floor"].items():
        if name not in METRICS or metric_floor <= 0:
            raise ConfigError(f"scale_floor: unknown metric {name!r} or non-positive floor")
    for cat, spec in cfg["categories"].items():
        if cat not in EQUIPMENT_CATEGORIES:
            raise ConfigError(f"unknown category {cat!r}")
        for metric, dets in (spec.get("metrics") or {}).items():
            if metric not in METRICS:
                raise ConfigError(f"{cat}: unknown metric {metric!r}")
            unknown = [d for d in dets if d not in UNIVARIATE]
            if unknown:
                raise ConfigError(f"{cat}.{metric}: unknown detectors {unknown}")
            if "transitions" in dets and METRICS[metric].kind != "state":
                raise ConfigError(f"{cat}.{metric}: 'transitions' is only for state metrics")
            if METRICS[metric].kind == "state" and set(dets) - {"transitions"}:
                raise ConfigError(f"{cat}.{metric}: state metrics only support 'transitions'")
        for key in ("mains_outage_expected", "recharge_expected", "multivariate"):
            for metric in spec.get(key) or []:
                if metric not in (spec.get("metrics") or {}):
                    raise ConfigError(f"{cat}.{key}: {metric!r} is not one of the category's metrics")
    for metric, dets in (cfg.get("detector_overrides") or {}).items():
        if metric not in METRICS or not set(dets) <= set(UNIVARIATE):
            raise ConfigError(f"detector_overrides: unknown metric {metric!r} or detector in {list(dets)}")
    h = cfg["health"]
    b = h["bands"]
    if not 100 >= b["healthy"] > b["warning"] > b["degraded"] > 0:
        raise ConfigError("health.bands must satisfy 100 >= healthy > warning > degraded > 0")
    if not 0 < h["min_coverage"] <= 1:
        raise ConfigError("health.min_coverage must be in (0, 1]")
    for cat, trends in (cfg["maintenance"].get("trends") or {}).items():
        if cat not in EQUIPMENT_CATEGORIES:
            raise ConfigError(f"maintenance.trends: unknown category {cat!r}")
        for metric, t in trends.items():
            if metric not in METRICS or t.get("direction") not in ("increasing", "decreasing"):
                raise ConfigError(f"maintenance.trends.{cat}.{metric}: unknown metric or direction")
            if not t.get("action"):
                raise ConfigError(f"maintenance.trends.{cat}.{metric}: an inspection action is required")


def metric_config(cfg: dict, category: str, metric: str) -> list[str] | None:
    """Detectors configured for this category/metric, or None if the metric is not monitored there."""
    return ((cfg["categories"].get(category) or {}).get("metrics") or {}).get(metric)


def detector_params(cfg: dict, name: str, metric: str | None = None) -> dict:
    p = dict(cfg["detectors"].get(name) or {})
    if metric:
        p.update(((cfg.get("detector_overrides") or {}).get(metric) or {}).get(name) or {})
    return p
