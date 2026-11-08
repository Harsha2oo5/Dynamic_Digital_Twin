"""Analytics API (/api/analytics/...): equipment health, anomalies, maintenance recommendations, configuration and
anomaly models. Live events (anomaly, health, recommendation) are pushed on the existing /ws/telemetry socket.

Scores are threshold units and health is an index of recent abnormal behaviour; neither is a failure probability.
"""

from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.analytics.config import ConfigError
from app.analytics.service import AnalyticsService, AnalyticsUnavailable
from app.telemetry.service import RANGES
from app.telemetry.store import NotFound

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def get_analytics(request: Request) -> AnalyticsService:
    svc: AnalyticsService | None = getattr(request.app.state, "analytics", None)
    if svc is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "analytics is not initialised")
    try:
        svc.require()
    except AnalyticsUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from None
    return svc


class StatusChange(BaseModel):
    by: str = Field(min_length=1, max_length=100)
    note: str | None = Field(None, max_length=2000)


class ConfigUpdate(BaseModel):
    content: str = Field(min_length=10, description="Complete YAML configuration")
    by: str = Field(min_length=1, max_length=100)
    note: str | None = Field(None, max_length=2000)


def _window(range_: str) -> datetime:
    if range_ not in RANGES:
        raise HTTPException(422, f"range must be one of {list(RANGES)}")
    return datetime.now(UTC) - timedelta(seconds=RANGES[range_])


# --- health ---------------------------------------------------------------------------------------------
@router.get("/status")
def analytics_status(request: Request):
    svc: AnalyticsService | None = getattr(request.app.state, "analytics", None)
    if svc is None or not svc.started:
        return {"available": False, "error": svc.error if svc else "analytics is not initialised"}
    return jsonable_encoder({"available": True, "config_version": svc.config_version, "models": svc.loaded_models,
                             "engine": svc.engine.benchmark()})


@router.get("/health/summary")
def health_summary(svc: AnalyticsService = Depends(get_analytics)):
    return jsonable_encoder(svc.summary())


@router.get("/health/equipment")
def health_list(category: str | None = None,
                status_: Literal["healthy", "warning", "degraded", "critical", "unknown"] | None = Query(None, alias="status"),
                svc: AnalyticsService = Depends(get_analytics)):
    with svc._lock:
        rows = list(svc.health.values())
    rows = [r for r in rows if (not category or r["category"] == category) and (not status_ or r["status"] == status_)]
    return jsonable_encoder(sorted(rows, key=lambda r: (r["score"] is None, r["score"] if r["score"] is not None else 0)))


@router.get("/health/equipment/{equipment_id}")
def health_one(equipment_id: str, svc: AnalyticsService = Depends(get_analytics)):
    r = svc.health_of(equipment_id)
    if r is None:
        raise HTTPException(404, f"equipment {equipment_id} not found")
    return jsonable_encoder(svc.health_out(r))


@router.get("/health/equipment/{equipment_id}/history")
def health_history(equipment_id: str, range_: str = Query("24h", alias="range"),
                   svc: AnalyticsService = Depends(get_analytics)):
    rows = svc.store.health_history(equipment_id, _window(range_))
    return jsonable_encoder([{"at": r.at, "score": r.score, "status": r.status, "coverage": r.coverage,
                              "data_quality": r.data_quality} for r in rows])


# --- anomalies ------------------------------------------------------------------------------------------
@router.get("/anomalies")
def anomalies(equipment_id: str | None = None, sensor_id: str | None = None, category: str | None = None,
              severity: Literal["warning", "critical"] | None = None,
              status_: Literal["open", "acknowledged", "resolved"] | None = Query(None, alias="status"),
              metric: str | None = None, detector: str | None = None, active: bool | None = None,
              data_source: Literal["physical", "simulated"] | None = None,
              start: datetime | None = None, end: datetime | None = None,
              limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
              svc: AnalyticsService = Depends(get_analytics)):
    total, rows = svc.store.anomalies_page(limit, offset, equipment_id=equipment_id, sensor_id=sensor_id,
                                           category=category, severity=severity, status=status_, metric=metric,
                                           detector=detector, active=active, data_source=data_source,
                                           start=start, end=end)
    return {"total": total, "limit": limit, "offset": offset, "items": [svc.anomaly_out(a) for a in rows]}


@router.get("/anomalies/{anomaly_id}")
def anomaly(anomaly_id: int, svc: AnalyticsService = Depends(get_analytics)):
    a = svc.store.get_anomaly(anomaly_id)
    if a is None:
        raise HTTPException(404, f"anomaly {anomaly_id} not found")
    return svc.anomaly_out(a)


def _status_change(svc: AnalyticsService, anomaly_id: int, new: str, body: StatusChange) -> dict:
    try:
        a = svc.store.set_anomaly_status(anomaly_id, new, body.by, body.note)
    except NotFound as exc:
        raise HTTPException(404, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    svc.engine.operator_resolved(a)
    out = svc.anomaly_out(a)
    svc.t.hub.publish({"type": "anomaly", "event": f"anomaly_{new}", "anomaly": out})
    return out


@router.post("/anomalies/{anomaly_id}/ack")
def acknowledge(anomaly_id: int, body: StatusChange, svc: AnalyticsService = Depends(get_analytics)):
    return _status_change(svc, anomaly_id, "acknowledged", body)


@router.post("/anomalies/{anomaly_id}/resolve")
def resolve(anomaly_id: int, body: StatusChange, svc: AnalyticsService = Depends(get_analytics)):
    return _status_change(svc, anomaly_id, "resolved", body)


# --- maintenance ---------------------------------------------------------------------------------------
@router.get("/recommendations")
def recommendations(equipment_id: str | None = None, category: str | None = None,
                    status_: Literal["active", "resolved"] | None = Query("active", alias="status"),
                    evidence_level: Literal["low", "medium", "high"] | None = None,
                    limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                    svc: AnalyticsService = Depends(get_analytics)):
    total, rows = svc.store.recommendations_page(limit, offset, equipment_id=equipment_id, category=category,
                                                 status=status_, evidence_level=evidence_level)
    return jsonable_encoder({"total": total, "limit": limit, "offset": offset, "items": [svc.rec_out(r) for r in rows]})


@router.post("/recommendations/{rec_id}/ack")
def acknowledge_recommendation(rec_id: int, body: StatusChange, svc: AnalyticsService = Depends(get_analytics)):
    try:
        return jsonable_encoder(svc.rec_out(svc.store.acknowledge_recommendation(rec_id, body.by)))
    except NotFound as exc:
        raise HTTPException(404, str(exc)) from None


@router.post("/maintenance/run")
async def run_maintenance(svc: AnalyticsService = Depends(get_analytics)):
    changed = await run_in_threadpool(svc.run_maintenance)
    await run_in_threadpool(svc.refresh_health)
    return {"updated_recommendations": len(changed)}


# --- equipment details (vision + telemetry + analytics in one place) -------------------------------------
@router.get("/equipment/{equipment_id}/details")
def equipment_details(equipment_id: str, svc: AnalyticsService = Depends(get_analytics)):
    t = svc.t
    eq = t.store.get_equipment(equipment_id)
    if eq is None:
        raise HTTPException(404, f"equipment {equipment_id} not found")
    now = datetime.now(UTC)
    _, anomalies_ = svc.store.anomalies_page(50, 0, equipment_id=equipment_id)
    health = svc.health_of(equipment_id)
    return jsonable_encoder({
        "equipment": t.equipment_out(eq, now),
        "sensors": [t.sensor_out(s, now) for s in t.store.list_sensors() if s.equipment_id == equipment_id],
        "latest": t.latest(equipment_id=equipment_id),
        "health": svc.health_out(health) if health else None,
        "health_history": [{"at": r.at, "score": r.score, "status": r.status}
                           for r in svc.store.health_history(equipment_id, now - timedelta(days=7))],
        "anomalies": [svc.anomaly_out(a) for a in anomalies_],
        "recommendations": [svc.rec_out(r) for r in svc.store.active_recommendations(equipment_id)],
        "threshold_alerts": [t.alert_out(a) for a in t.alerts.active() if a.equipment_id == equipment_id],
        "visual": {"links": t.equipment_out(eq, now)["visual_links"],
                   "note": "Visual observations are session-scoped tracks linked to this equipment manually."},
    })


# --- configuration and models ---------------------------------------------------------------------------
@router.get("/config")
def get_config(svc: AnalyticsService = Depends(get_analytics)):
    c = svc.store.active_config()
    return jsonable_encoder({"version": c.version, "config_hash": c.config_hash, "created_at": c.created_at,
                             "created_by": c.created_by, "note": c.note, "content": c.content, "parsed": svc.cfg})


@router.get("/config/versions")
def config_versions(svc: AnalyticsService = Depends(get_analytics)):
    return jsonable_encoder([{"version": c.version, "config_hash": c.config_hash, "active": c.active,
                              "created_at": c.created_at, "created_by": c.created_by, "note": c.note}
                             for c in svc.store.configs()])


@router.put("/config")
async def put_config(body: ConfigUpdate = Body(...), svc: AnalyticsService = Depends(get_analytics)):
    try:
        version = await run_in_threadpool(svc.update_config, body.content, body.by, body.note)
    except ConfigError as exc:
        raise HTTPException(422, f"invalid configuration: {exc}") from None
    return {"version": version, "note": "feature windows restart; metrics are re-warmed from stored history"}


@router.get("/models")
def models(svc: AnalyticsService = Depends(get_analytics)):
    return jsonable_encoder([{c: getattr(m, c) for c in ("id", "category", "data_source", "algorithm", "version",
                                                          "active", "artifact_path", "trained_at", "config_hash",
                                                          "features", "params", "thresholds", "training", "evaluation")}
                             for m in svc.store.models()])


@router.post("/models/{model_id}/activate")
def activate_model(model_id: int, svc: AnalyticsService = Depends(get_analytics)):
    try:
        svc.store.activate_model(model_id)
    except NotFound as exc:
        raise HTTPException(404, str(exc)) from None
    return {"loaded": svc.reload_models()}


@router.post("/models/reload")
def reload_models(svc: AnalyticsService = Depends(get_analytics)):
    return {"loaded": svc.reload_models()}
