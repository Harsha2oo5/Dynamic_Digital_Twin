"""Telemetry REST API (/api/telemetry/...) and the live WebSocket (/ws/telemetry).

WebSocket protocol (server -> client only; the client may send {"type": "ping"}):
  {"type": "snapshot", "summary": ..., "sensors": [...], "latest": [...], "alerts": [...]}   on connect
  {"type": "batch", "events": [ {"type": "measurements", "items": [...]},
                                {"type": "sensor_status", "sensor_id": ..., "status": ...},
                                {"type": "alert", "alert": {...}} ], "dropped": n}         as data arrives (<= 4/s)
"""

import asyncio
import hmac
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Security, WebSocket, WebSocketDisconnect, status
from fastapi.encoders import jsonable_encoder
from fastapi.security import APIKeyHeader
from starlette.concurrency import run_in_threadpool

from app.telemetry.catalog import METRICS, catalogue
from app.telemetry.schemas import (
    AlertAck,
    AlertOut,
    AssociationIn,
    EquipmentCreate,
    EquipmentOut,
    EquipmentUpdate,
    IngestBatch,
    IngestMessage,
    IngestResult,
    LatestReading,
    SensorCreate,
    SensorOut,
    SensorUpdate,
    SimulatorScenario,
    SimulatorStart,
    VisualLinkIn,
)
from app.telemetry.service import RANGES, TelemetryService, TelemetryUnavailable
from app.telemetry.simulator import SimulatorError
from app.telemetry.store import Conflict, NotFound

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])
ws_router = APIRouter()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_telemetry(request: Request) -> TelemetryService:
    svc: TelemetryService | None = getattr(request.app.state, "telemetry", None)
    if svc is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "telemetry is not initialised")
    try:
        svc.require()
    except TelemetryUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from None
    return svc


def require_ingest_key(request: Request, key: str | None = Security(api_key_header)) -> None:
    configured = request.app.state.telemetry.settings.telemetry_ingest_api_key if getattr(
        request.app.state, "telemetry", None) else None
    if configured is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "HTTP ingestion is closed: set TELEMETRY_INGEST_API_KEY on the server")
    if key is None or not hmac.compare_digest(key.encode(), configured.get_secret_value().encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid X-API-Key")


def _errors(fn):
    try:
        return fn()
    except NotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from None
    except Conflict as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None
    except SimulatorError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None


def _window(range_: str | None, start: datetime | None, end: datetime | None) -> tuple[datetime, datetime]:
    now = datetime.now(UTC)
    if start is not None or end is not None:
        end = (end or now).astimezone(UTC) if (end or now).tzinfo else (end or now).replace(tzinfo=UTC)
        start = (start or end - timedelta(hours=1))
        start = start.astimezone(UTC) if start.tzinfo else start.replace(tzinfo=UTC)
        if start >= end:
            raise HTTPException(422, "start must be before end")
        return start, end
    seconds = RANGES.get(range_ or "1h")
    if seconds is None:
        raise HTTPException(422, f"range must be one of {list(RANGES)}")
    return now - timedelta(seconds=seconds), now


def _page(total: int, limit: int, offset: int, items: list) -> dict:
    return {"total": total, "limit": limit, "offset": offset, "items": items}


# --- status / catalogue ----------------------------------------------------------------------------
@router.get("/status")
def telemetry_status(request: Request):
    svc: TelemetryService | None = getattr(request.app.state, "telemetry", None)
    if svc is None:
        return {"available": False, "error": "telemetry is not initialised"}
    return jsonable_encoder(svc.info())


@router.get("/metrics")
def metrics():
    return catalogue()


@router.get("/summary")
def summary(svc: TelemetryService = Depends(get_telemetry)):
    return jsonable_encoder(svc.summary())


# --- equipment ---------------------------------------------------------------------------------------
@router.get("/equipment", response_model=list[EquipmentOut])
def list_equipment(category: str | None = None, data_source: Literal["physical", "simulated"] | None = None,
                   svc: TelemetryService = Depends(get_telemetry)):
    now = datetime.now(UTC)
    return [svc.equipment_out(e, now) for e in svc.store.list_equipment(category, data_source)]


@router.post("/equipment", response_model=EquipmentOut, status_code=201)
def create_equipment(body: EquipmentCreate, svc: TelemetryService = Depends(get_telemetry)):
    if body.data_source == "simulated" or body.equipment_id.upper().startswith("SIM-"):
        raise HTTPException(422, "simulated equipment is created by the simulator only (IDs starting with SIM-)")
    eq = _errors(lambda: svc.store.create_equipment(svc.site_id, body.model_dump()))
    return svc.equipment_out(eq)


@router.get("/equipment/{equipment_id}", response_model=EquipmentOut)
def get_equipment(equipment_id: str, svc: TelemetryService = Depends(get_telemetry)):
    eq = svc.store.get_equipment(equipment_id)
    if eq is None:
        raise HTTPException(404, f"equipment {equipment_id} not found")
    return svc.equipment_out(eq)


@router.patch("/equipment/{equipment_id}", response_model=EquipmentOut)
def update_equipment(equipment_id: str, body: EquipmentUpdate, svc: TelemetryService = Depends(get_telemetry)):
    eq = _errors(lambda: svc.store.update_equipment(equipment_id, body.model_dump(exclude_unset=True)))
    return svc.equipment_out(eq)


@router.delete("/equipment/{equipment_id}", status_code=204)
def delete_equipment(equipment_id: str, svc: TelemetryService = Depends(get_telemetry)):
    sensors = [i.sensor_id for i in svc.pipeline.sensors.values() if i.equipment_id == equipment_id]
    _errors(lambda: svc.store.delete_equipment(equipment_id))
    for sid in sensors:
        svc.sensor_changed(sid)


@router.get("/equipment/{equipment_id}/telemetry")
def equipment_telemetry(equipment_id: str, svc: TelemetryService = Depends(get_telemetry)):
    eq = svc.store.get_equipment(equipment_id)
    if eq is None:
        raise HTTPException(404, f"equipment {equipment_id} not found")
    now = datetime.now(UTC)
    sensors = [svc.sensor_out(s, now) for s in svc.store.list_sensors() if s.equipment_id == equipment_id]
    alerts = [svc.alert_out(a) for a in svc.alerts.active() if a.equipment_id == equipment_id]
    return jsonable_encoder({"equipment": svc.equipment_out(eq, now), "sensors": sensors,
                             "latest": svc.latest(equipment_id=equipment_id), "active_alerts": alerts})


@router.post("/equipment/{equipment_id}/visual-links", status_code=201)
def add_visual_link(equipment_id: str, body: VisualLinkIn, svc: TelemetryService = Depends(get_telemetry)):
    """Explicitly link a visual observation (inventory object) to this equipment. Never done automatically."""
    link = _errors(lambda: svc.store.add_visual_link(equipment_id, body.model_dump()))
    return jsonable_encoder({"id": link.id, "equipment_id": equipment_id, "object_id": link.object_id,
                             "linked_at": link.linked_at, "note": "tracking IDs are session-scoped; this link "
                             "records an observation, not a permanent identity"})


@router.delete("/equipment/{equipment_id}/visual-links/{link_id}", status_code=204)
def delete_visual_link(equipment_id: str, link_id: int, svc: TelemetryService = Depends(get_telemetry)):
    _errors(lambda: svc.store.delete_visual_link(link_id))


# --- sensors ----------------------------------------------------------------------------------------
@router.get("/sensors")
def list_sensors(sensor_type: str | None = None, category: str | None = None, equipment_id: str | None = None,
                 protocol: str | None = None, data_source: Literal["physical", "simulated"] | None = None,
                 status_: Literal["unknown", "online", "stale", "offline", "disabled"] | None = Query(None, alias="status"),
                 limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
                 svc: TelemetryService = Depends(get_telemetry)):
    now = datetime.now(UTC)
    rows = [svc.sensor_out(s, now) for s in svc.store.list_sensors()]
    for key, val in (("sensor_type", sensor_type), ("category", category), ("equipment_id", equipment_id),
                     ("protocol", protocol), ("data_source", data_source), ("status", status_)):
        if val is not None:
            rows = [r for r in rows if r[key] == val]
    return jsonable_encoder(_page(len(rows), limit, offset, rows[offset:offset + limit]))


@router.post("/sensors", response_model=SensorOut, status_code=201)
def create_sensor(body: SensorCreate, svc: TelemetryService = Depends(get_telemetry)):
    if body.protocol == "simulator" or body.data_source == "simulated" or body.sensor_id.upper().startswith("SIM-"):
        raise HTTPException(422, "simulated sensors are created by the simulator only (IDs starting with SIM-)")
    if body.protocol == "modbus":
        raise HTTPException(422, "the Modbus adapter is not implemented yet (needs device register maps)")
    sensor = _errors(lambda: svc.store.create_sensor(svc.site_id, body.model_dump()))
    svc.sensor_changed(sensor.sensor_id)
    return svc.sensor_out(sensor)


@router.get("/sensors/{sensor_id}", response_model=SensorOut)
def get_sensor(sensor_id: str, svc: TelemetryService = Depends(get_telemetry)):
    s = svc.store.get_sensor(sensor_id)
    if s is None:
        raise HTTPException(404, f"sensor {sensor_id} not found")
    return svc.sensor_out(s)


@router.patch("/sensors/{sensor_id}", response_model=SensorOut)
def update_sensor(sensor_id: str, body: SensorUpdate, svc: TelemetryService = Depends(get_telemetry)):
    s = _errors(lambda: svc.store.update_sensor(sensor_id, body.model_dump(exclude_unset=True)))
    svc.sensor_changed(sensor_id)
    return svc.sensor_out(s)


@router.put("/sensors/{sensor_id}/equipment", response_model=SensorOut)
def associate_sensor(sensor_id: str, body: AssociationIn, svc: TelemetryService = Depends(get_telemetry)):
    """Explicitly attach a sensor to an equipment item (or detach it with null). Earlier measurements keep the
    equipment they were recorded for."""
    s = _errors(lambda: svc.store.associate(sensor_id, body.equipment_id))
    svc.sensor_changed(sensor_id)
    return svc.sensor_out(s)


@router.delete("/sensors/{sensor_id}", status_code=204)
def delete_sensor(sensor_id: str, svc: TelemetryService = Depends(get_telemetry)):
    s = svc.store.get_sensor(sensor_id)
    if s is not None and s.data_source == "simulated" and svc.simulator and svc.simulator.running:
        raise HTTPException(409, "stop the simulator before deleting simulated sensors")
    _errors(lambda: svc.store.delete_sensor(sensor_id))
    svc.sensor_changed(sensor_id, deleted=True)


# --- ingestion --------------------------------------------------------------------------------------
@router.post("/ingest", response_model=IngestResult, dependencies=[Depends(require_ingest_key)])
async def ingest(body: IngestMessage, svc: TelemetryService = Depends(get_telemetry)):
    return await run_in_threadpool(svc.pipeline.ingest, body, "http")


@router.post("/ingest/batch", response_model=IngestResult, dependencies=[Depends(require_ingest_key)])
async def ingest_batch(body: IngestBatch, svc: TelemetryService = Depends(get_telemetry)):
    return await run_in_threadpool(svc.pipeline.ingest_many, body.messages, "http")


# --- readings ---------------------------------------------------------------------------------------
@router.get("/latest", response_model=list[LatestReading])
def latest(sensor_id: str | None = None, equipment_id: str | None = None, metric: str | None = None,
           category: str | None = None, svc: TelemetryService = Depends(get_telemetry)):
    return svc.latest(sensor_id, equipment_id, metric, category)


@router.get("/history")
def history(metric: str, sensor_id: str | None = None, equipment_id: str | None = None,
            range_: str | None = Query("1h", alias="range"), start: datetime | None = None,
            end: datetime | None = None, max_points: int = Query(300, ge=10, le=2000),
            svc: TelemetryService = Depends(get_telemetry)):
    if metric not in METRICS:
        raise HTTPException(422, f"unknown metric {metric!r}")
    t0, t1 = _window(range_, start, end)
    return jsonable_encoder({"metric": metric, "start": t0, "end": t1,
                             "series": svc.history(metric, t0, t1, sensor_id, equipment_id, max_points)})


@router.get("/measurements")
def measurements(sensor_id: str | None = None, equipment_id: str | None = None, metric: str | None = None,
                 quality: Literal["good", "missing", "invalid", "out_of_order"] | None = None,
                 data_source: Literal["physical", "simulated"] | None = None,
                 start: datetime | None = None, end: datetime | None = None,
                 limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
                 svc: TelemetryService = Depends(get_telemetry)):
    total, rows = svc.store.measurements_page(limit, offset, sensor_id=sensor_id, equipment_id=equipment_id,
                                              metric=metric, quality=quality, data_source=data_source,
                                              start=start, end=end)
    items = [{"id": m.id, "sensor_id": m.sensor_id, "equipment_id": m.equipment_id, "metric": m.metric,
              "value": m.value, "unit": m.unit, "ts": m.ts, "received_at": m.received_at, "data_source": m.data_source,
              "protocol": m.protocol, "quality": m.quality, "quality_note": m.quality_note,
              "message_id": m.message_id} for m in rows]
    return jsonable_encoder(_page(total, limit, offset, items))


@router.get("/availability")
def availability(range_: str | None = Query("24h", alias="range"), svc: TelemetryService = Depends(get_telemetry)):
    t0, t1 = _window(range_, None, None)
    return jsonable_encoder({"start": t0, "end": t1, "sensors": svc.availability(t0, t1)})


@router.get("/sensors/{sensor_id}/status-history")
def status_history(sensor_id: str, range_: str | None = Query("24h", alias="range"),
                   svc: TelemetryService = Depends(get_telemetry)):
    t0, _ = _window(range_, None, None)
    return jsonable_encoder([{"status": e.status, "at": e.at} for e in svc.store.status_events(t0, sensor_id)])


# --- alerts ------------------------------------------------------------------------------------------
@router.get("/alerts")
def alerts(state: Literal["active", "cleared"] | None = None, severity: Literal["warning", "critical"] | None = None,
           equipment_id: str | None = None, sensor_id: str | None = None, acknowledged: bool | None = None,
           limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
           svc: TelemetryService = Depends(get_telemetry)):
    total, rows = svc.store.alerts_page(limit, offset, state, severity, equipment_id, sensor_id, acknowledged)
    return _page(total, limit, offset, [svc.alert_out(a) for a in rows])


@router.post("/alerts/{alert_id}/ack", response_model=AlertOut)
def acknowledge(alert_id: int, body: AlertAck, svc: TelemetryService = Depends(get_telemetry)):
    a = _errors(lambda: svc.store.acknowledge(alert_id, body.acknowledged_by, body.note))
    for active in svc.alerts.active():  # keep the in-memory copy in step
        if active.id == a.id:
            active.acknowledged, active.acknowledged_at = a.acknowledged, a.acknowledged_at
            active.acknowledged_by, active.ack_note = a.acknowledged_by, a.ack_note
    out = svc.alert_out(a)
    svc.hub.publish({"type": "alert", "alert": out})
    return out


@router.get("/alerts/rules")
def alert_rules(svc: TelemetryService = Depends(get_telemetry)):
    c = svc.alerts.config
    return {"rules": [r.__dict__ | {"categories": list(r.categories)} for r in c.rules],
            "communication": {"enabled": c.communication_enabled, "critical_after_s": c.communication_critical_after_s}}


# --- simulator ---------------------------------------------------------------------------------------
@router.get("/simulator")
def simulator_status(svc: TelemetryService = Depends(get_telemetry)):
    return jsonable_encoder(svc.simulator.status())


@router.post("/simulator/start")
async def simulator_start(body: SimulatorStart, svc: TelemetryService = Depends(get_telemetry)):
    st = await run_in_threadpool(lambda: _errors(lambda: svc.simulator.start(body.interval_scale, body.backfill_hours,
                                                                              body.seed)))
    return jsonable_encoder(st)


@router.post("/simulator/stop")
async def simulator_stop(svc: TelemetryService = Depends(get_telemetry)):
    return jsonable_encoder(await run_in_threadpool(svc.simulator.stop))


@router.post("/simulator/scenario")
def simulator_scenario(body: SimulatorScenario, svc: TelemetryService = Depends(get_telemetry)):
    return jsonable_encoder(_errors(lambda: svc.simulator.scenario(body.scenario, body.duration_s, body.sensor_id)))


# --- WebSocket ---------------------------------------------------------------------------------------
@ws_router.websocket("/ws/telemetry")
async def telemetry_ws(ws: WebSocket):
    await ws.accept()
    svc: TelemetryService | None = getattr(ws.app.state, "telemetry", None)
    if svc is None or not svc.started:
        await ws.send_json({"type": "error", "message": (svc.error if svc else None) or "telemetry unavailable"})
        await ws.close()
        return
    svc.hub.bind_loop(asyncio.get_running_loop())
    sub = svc.hub.subscribe()
    receiver: asyncio.Task | None = None
    try:
        snapshot = await run_in_threadpool(lambda: {
            "type": "snapshot", "summary": svc.summary(),
            "sensors": [svc.sensor_out(s) for s in svc.store.list_sensors()], "latest": svc.latest(),
            "alerts": [svc.alert_out(a) for a in svc.alerts.active()]})
        await ws.send_text(json.dumps(jsonable_encoder(snapshot)))
        receiver = asyncio.create_task(ws.receive_text())
        while True:
            getter = asyncio.create_task(sub.queue.get())
            done, _ = await asyncio.wait({getter, receiver}, return_when=asyncio.FIRST_COMPLETED)
            if receiver in done:
                getter.cancel()
                receiver.result()  # raises WebSocketDisconnect when the client left
                receiver = asyncio.create_task(ws.receive_text())
                continue
            events = [getter.result()]
            await asyncio.sleep(0.25)  # batch whatever else arrives (at most 4 messages per second)
            while not sub.queue.empty() and len(events) < 500:
                events.append(sub.queue.get_nowait())
            await ws.send_text(json.dumps(jsonable_encoder({"type": "batch", "events": events,
                                                            "dropped": sub.dropped})))
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        logger.exception("Telemetry WebSocket failed")
    finally:
        if receiver is not None and not receiver.done():
            receiver.cancel()
        svc.hub.unsubscribe(sub)
