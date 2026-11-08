"""Trend-based predictive maintenance. No failure dates are predicted: there is no historical failure data.

For every equipment metric configured under maintenance.trends:
  1. hourly medians of good readings over `window_h` (default 7 days); with `only_when_mains`, hours during a mains
     outage or the recharge period after it are excluded (discharge is expected there)
  2. Theil-Sen slope (robust to outliers) with its 95 % confidence interval, and Kendall's tau test for a monotonic
     trend (a Mann-Kendall-type test)
  3. a degradation trend is reported when the slope points in the configured direction, exceeds
     `min_slope_per_day`, its confidence interval excludes zero and p < 0.05
Evidence level (how much the data supports the trend, not a probability of failure):
  high    p < 0.001, span >= 5 days, at least 96 hourly points
  medium  p < 0.01,  span >= 3 days
  low     otherwise (span >= min_span_h)
Hourly medians are autocorrelated, so p-values are optimistic; the slope and span requirements guard against that.
Anomaly frequency: >= `min_last_24h` anomalies in the last 24 h and `ratio` times the earlier daily average.
"""

import math
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from scipy import stats

from app.analytics.models import MaintenanceRecommendation
from app.telemetry.catalog import METRICS


def hourly_medians(rows: list[tuple], exclude_hours: set[int] | None = None, min_per_hour: int = 3):
    by_hour: dict[int, list[float]] = defaultdict(list)
    for _sid, _eq, ts, value, _src in rows:
        h = int(ts.timestamp() // 3600)
        if exclude_hours and h in exclude_hours:
            continue
        by_hour[h].append(value)
    hours = sorted(h for h, v in by_hour.items() if len(v) >= min_per_hour)
    return np.array(hours, float), np.array([float(np.median(by_hour[h])) for h in hours])


def outage_hours(mains_rows: list[tuple], after_s: float) -> set[int]:
    out = set()
    extra = int(math.ceil(after_s / 3600))
    for _sid, _eq, ts, value, _src in mains_rows:
        if value < 0.5:
            h = int(ts.timestamp() // 3600)
            out.update(range(h, h + extra + 1))
    return out


def trend(hours: np.ndarray, values: np.ndarray) -> dict | None:
    if len(hours) < 3 or np.ptp(values) == 0:
        return None
    days = (hours - hours[0]) / 24.0
    slope, intercept, lo, hi = stats.theilslopes(values, days, alpha=0.95)
    tau, p = stats.kendalltau(days, values)
    # start/end of the FITTED line (raw first/last hourly medians swing with the daily cycle)
    return {"slope_per_day": float(slope), "ci_low": float(lo), "ci_high": float(hi), "tau": float(tau),
            "p_value": float(p), "n": int(len(hours)), "span_h": float(hours[-1] - hours[0]),
            "first": float(intercept), "last": float(intercept + slope * days[-1]), "intercept": float(intercept)}


def evidence_level(t: dict) -> str:
    if t["p_value"] < 0.001 and t["span_h"] >= 120 and t["n"] >= 96:
        return "high"
    if t["p_value"] < 0.01 and t["span_h"] >= 72:
        return "medium"
    return "low"


def analyse_equipment(eq, sensor_metrics: set[str], series, cfg: dict, now: datetime, mains_rows: list[tuple],
                      anomaly_times: list[datetime], related_anomalies) -> tuple[list[dict], bool]:
    """Candidate recommendations for one equipment and whether any trend could be assessed (enough data)."""
    m_cfg = cfg["maintenance"]
    start = now - timedelta(hours=m_cfg["window_h"])
    out, assessed = [], False
    excl = outage_hours(mains_rows, cfg.get("context", {}).get("after_mains_restored_s", 0))
    for metric, rule in ((m_cfg.get("trends") or {}).get(eq.category) or {}).items():
        if metric not in sensor_metrics:
            continue
        rows = series(metric, start, now, equipment_id=eq.equipment_id)
        hours, vals = hourly_medians(rows, excl if rule.get("only_when_mains") else None)
        if len(hours) < m_cfg["min_points"] or (hours[-1] - hours[0]) < m_cfg["min_span_h"]:
            continue
        assessed = True
        t = trend(hours, vals)
        if t is None:
            continue
        sign = 1 if rule["direction"] == "increasing" else -1
        s = sign * t["slope_per_day"]
        ci_ok = (t["ci_low"] > 0) if sign > 0 else (t["ci_high"] < 0)
        if not (s >= rule["min_slope_per_day"] and ci_ok and t["p_value"] < 0.05 and sign * t["tau"] > 0):
            continue
        m = METRICS[metric]
        unit = "" if m.unit in ("state", "ratio") else f" {m.unit}"
        step = max(1, len(hours) // 168)
        rel = [a.id for a in related_anomalies if a.metric == metric][:20]
        out.append({
            "condition_key": f"trend:{metric}:{rule['direction']}", "metric": metric, "action": rule["action"],
            "evidence_level": evidence_level(t),
            "condition": (f"{m.label} {rule['direction']} by {t['slope_per_day']:+.3g}{unit}/day over the last "
                          f"{t['span_h'] / 24:.1f} days (trend line {t['first']:.3g} → {t['last']:.3g}{unit}). "
                          "Trend-based indication for inspection; no failure date is predicted."),
            "evidence": {"method": "Theil-Sen slope on hourly medians + Kendall tau test", **{k: round(v, 5) if
                         isinstance(v, float) else v for k, v in t.items()},
                         "min_slope_per_day": rule["min_slope_per_day"], "excluded_outage_hours":
                         bool(rule.get("only_when_mains")),
                         "series": [{"t": datetime.fromtimestamp(h * 3600, now.tzinfo).isoformat(), "v": round(v, 4)}
                                    for h, v in zip(hours[::step], vals[::step])],
                         "related_anomalies": rel}})
    fq = m_cfg.get("anomaly_frequency")
    if fq and anomaly_times:
        last24 = sum(1 for t in anomaly_times if t >= now - timedelta(hours=24))
        earlier = [t for t in anomaly_times if t < now - timedelta(hours=24)]
        days_prev = max(1.0, min(6.0, (now - timedelta(hours=24) - min(anomaly_times)).total_seconds() / 86400)) \
            if earlier else 1.0
        prev_rate = len(earlier) / days_prev
        if last24 >= fq["min_last_24h"] and last24 >= fq["ratio"] * max(prev_rate, 0.5):
            out.append({"condition_key": "anomaly_frequency", "metric": None, "action": fq["action"],
                         "evidence_level": "medium" if earlier and days_prev >= 3 else "low",
                         "condition": (f"{last24} anomalies in the last 24 h vs {prev_rate:.1f}/day before: "
                                       "abnormal behaviour is becoming more frequent."),
                         "evidence": {"method": "anomaly count comparison", "last_24h": last24,
                                      "previous_daily_rate": round(prev_rate, 2), "previous_days": round(days_prev, 1),
                                      "related_anomalies": [a.id for a in related_anomalies][:20]}})
    return out, assessed


def apply(store, site_id: str, eq, candidates: list[dict], assessed: bool, now: datetime) -> list:
    """Create/update active recommendations; resolve ones whose evidence disappeared (only when data was assessed)."""
    active = {r.condition_key: r for r in store.active_recommendations(eq.equipment_id)}
    changed = []
    for c in candidates:
        r = active.pop(c["condition_key"], None)
        if r is None:
            r = MaintenanceRecommendation(site_id=site_id, equipment_id=eq.equipment_id, category=eq.category,
                                          status="active", data_source=eq.data_source, created_at=now, **c)
        else:
            for k, v in c.items():
                setattr(r, k, v)
        r.updated_at = now
        changed.append(store.save_recommendation(r))
    for key, r in active.items():
        if assessed or key == "anomaly_frequency":
            r.status, r.resolved_at, r.updated_at = "resolved", now, now
            changed.append(store.save_recommendation(r))
    return changed
