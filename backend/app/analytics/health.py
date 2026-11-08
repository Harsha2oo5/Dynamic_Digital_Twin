"""Explainable equipment health score (0-100, higher = healthier). It is an INDEX of recent abnormal behaviour,
not a probability of failure.

  score = 100 - sum(penalties), clamped to 0..100

  anomaly penalty  = base[severity] x 2^(-hours since last active / half_life)
                     + duration_penalty_per_h x min(hours active, 3) + recurrence_penalty x (occurrences - 1),
                     capped at max_anomaly_penalty per anomaly; anomalies of the last `lookback_h` hours count
  alert penalty    = active threshold alerts of the equipment (communication alerts excluded: they are about the
                     sensor, not the equipment)
  recommendation   = active maintenance recommendations by evidence level

  status: >= bands.healthy -> healthy, >= warning -> warning, >= degraded -> degraded, else critical
  unknown: the equipment has no sensors, or fewer than `min_coverage` of its configured metrics have fresh,
           warmed-up data. Then no score is given.

Data quality (sensor availability) is reported separately and never lowers the health score.
"""

from datetime import datetime

from app.telemetry.alerts import COMMUNICATION_RULE


def status_for(score: float | None, bands: dict) -> str:
    if score is None:
        return "unknown"
    if score >= bands["healthy"]:
        return "healthy"
    if score >= bands["warning"]:
        return "warning"
    if score >= bands["degraded"]:
        return "degraded"
    return "critical"


def anomaly_penalty(a, now: datetime, h: dict) -> float:
    last = now if a.condition_active else (a.ended_at or a.last_detected_at)
    age_h = max(0.0, (now - last).total_seconds() / 3600)
    if not a.condition_active and age_h > h["lookback_h"]:
        return 0.0
    base = h["anomaly_penalty"][a.severity] * 2 ** (-age_h / h["half_life_h"])
    active_h = ((a.ended_at or now) - a.first_detected_at).total_seconds() / 3600
    p = base + h["duration_penalty_per_h"] * min(max(active_h, 0.0), 3.0) + h["recurrence_penalty"] * (a.occurrences - 1)
    return min(p, h["max_anomaly_penalty"])


def compute(equipment_id: str, category: str, coverage: dict, anomalies: list, alerts: list, recommendations: list,
            data_quality: float | None, cfg: dict, now: datetime) -> dict:
    h = cfg["health"]
    components = []
    for a in anomalies:
        p = anomaly_penalty(a, now, h)
        if p > 0.05:
            components.append({"type": "anomaly", "id": a.id, "severity": a.severity, "penalty": round(p, 1),
                               "active": a.condition_active,
                               "detail": f"{a.metric or 'multivariate'} ({a.detector}), {a.occurrences}x"})
    for al in alerts:
        if al.rule_id == COMMUNICATION_RULE:
            continue
        components.append({"type": "alert", "id": al.id, "severity": al.severity,
                           "penalty": float(h["alert_penalty"][al.severity]), "active": True, "detail": al.rule_id})
    for r in recommendations:
        p = float(h["recommendation_penalty"].get(r.evidence_level, 0))
        if p:
            components.append({"type": "recommendation", "id": r.id, "severity": r.evidence_level, "penalty": p,
                               "active": True, "detail": r.condition_key})
    cov = coverage.get("coverage", 0.0)
    known = bool(coverage.get("provided")) and cov >= h["min_coverage"]
    score = round(max(0.0, 100.0 - sum(c["penalty"] for c in components)), 1) if known else None
    if not coverage.get("provided"):
        reason = "no sensors with configured metrics"
    elif not known:
        reason = (f"insufficient data: {len(coverage['usable'])} of {len(coverage['provided'])} monitored metrics have "
                  f"fresh, warmed-up data (need {int(h['min_coverage'] * 100)}%)")
    else:
        reason = None
    return {"equipment_id": equipment_id, "category": category, "score": score,
            "status": status_for(score, h["bands"]), "coverage": cov, "coverage_detail": coverage.get("detail", {}),
            "data_quality": None if data_quality is None else round(data_quality, 3),
            "components": sorted(components, key=lambda c: -c["penalty"]), "unknown_reason": reason,
            "at": now}
