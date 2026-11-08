# Sensor integration and telemetry monitoring (Milestone 2)

This milestone adds operational telemetry next to the vision system: a registry of sites, equipment and sensors,
ingestion over HTTP and MQTT, PostgreSQL storage, live updates over WebSocket, threshold alerts, and a simulator
for demonstrations without hardware.

> **No physical sensors are assumed.** Everything works with the built-in simulator, and every simulated value is
> labelled `simulated` in the database, the API and the dashboard. Physical sensors plug in through the same
> pipeline once they exist (see [What needs real hardware](#what-works-today-and-what-needs-hardware)).

## Contents
1. [Architecture](#architecture)
2. [Data model](#data-model)
3. [Telemetry message format](#telemetry-message-format)
4. [Ingestion rules](#ingestion-rules)
5. [Connection status](#connection-status-and-stale-data)
6. [Alerts](#alerts)
7. [Registering sensors](#registering-equipment-and-sensors)
8. [MQTT](#mqtt-integration)
9. [Simulator](#simulator)
10. [API](#api)
11. [Configuration](#configuration)
12. [Database setup](#database-setup-postgresql)
13. [Testing](#testing)
14. [What works today and what needs hardware](#what-works-today-and-what-needs-hardware)
15. [Adding a protocol](#adding-a-protocol)

## Architecture

```
 sensor devices / simulator
     │  HTTP  POST /api/telemetry/ingest  (X-API-Key)
     │  MQTT  bts/<site>/<sensor>/telemetry  (paho-mqtt, reconnect with backoff)
     │  simulator  (in-process, protocol "simulator")
     │  Modbus  (interface only)
     ▼
 adapters ──► IngestionPipeline ─► registry check · channel check · duplicates · clock · unit normalisation
                     │                plausibility · ordering
                     ├─► TelemetryStore (SQLAlchemy) ─► PostgreSQL  (SQLite fallback for development)
                     ├─► latest-value cache
                     ├─► AlertEngine (thresholds.yaml) ─► alerts table
                     └─► TelemetryHub ─► WS /ws/telemetry ─► React dashboard
 watchdog thread (every 2 s): connection state online/stale/offline → status events, communication alerts,
                              last-seen flush, hourly retention purge
```

Code: `backend/app/telemetry/` (`catalog`, `models`, `db`, `migrations/`, `schemas`, `store`, `ingest`, `alerts`,
`hub`, `adapters`, `simulator`, `service`) and `backend/app/api/telemetry_routes.py`.

Design decisions:
- **Telemetry is independent of vision.** If the database is unreachable, only `/api/telemetry/*` answers 503 with
  the reason. Detection, tracking, the inventory and model management keep working.
- **Stable equipment identity.** Equipment IDs such as `BAT-001` are assigned by operators and stored in
  PostgreSQL. They are **not** ByteTrack IDs, which only exist inside one stream or video. A visual observation can
  be linked to equipment only explicitly (`POST /api/telemetry/equipment/{id}/visual-links`); nothing is matched
  automatically by class name or position.
- **One pipeline for every protocol.** Adapters only parse their transport; validation and storage are shared.
- **Threads, not processes.** MQTT runs in paho's network thread, the simulator and the watchdog each in one daemon
  thread. All three stop on application shutdown.

## Data model

```
sites ─┬─ equipment ─┬─ sensors ── measurements (time series)
       │             │          └─ sensor_status_events
       │             └─ equipment_visual_links (manual)
       └─ alerts
```

| Table | Key columns |
|---|---|
| `sites` | `site_id`, `name`, `location` |
| `equipment` | `equipment_id` (stable, e.g. `BAT-001`), `site_id`, `category` (one of the 7 classes), `name`, `zone`, `data_source` |
| `sensors` | `sensor_id`, `name`, `sensor_type`, `protocol` (http, mqtt, simulator), `equipment_id` (nullable), `category` (equipment category or `site`), `zone`, `data_source`, `enabled`, `metrics` (JSON list), `sampling_interval_s`, `stale_after_s`, `offline_after_s`, `config` (JSON, no secrets), `last_seen_at` |
| `measurements` | `id`, `sensor_id`, `equipment_id` (snapshot at ingest), `site_id`, `metric`, `value` (canonical unit, NULL if missing/invalid), `unit`, `ts`, `received_at`, `data_source`, `protocol`, `quality`, `quality_note`, `message_id`, `raw_value`, `raw_unit` |
| `sensor_status_events` | `sensor_id`, `status`, `at` |
| `alerts` | `rule_id`, `severity`, `state`, `sensor_id`, `equipment_id`, `metric`, `value`, `peak_value`, `threshold`, `message`, `data_source`, `opened_at`, `last_triggered_at`, `cleared_at`, `acknowledged`, `acknowledged_at`, `acknowledged_by`, `ack_note` |
| `equipment_visual_links` | `equipment_id`, `object_id` (inventory `source:track_id`), `class_name`, `linked_by`, `note`, `linked_at` |

Time-series indexing on `measurements`:
- unique `(sensor_id, metric, ts)`: the main per-sensor index, and the database-level duplicate guard;
- `(equipment_id, metric, ts)` and `(metric, ts)` for equipment and site queries;
- on PostgreSQL, a **BRIN** index on `ts`, which keeps range scans and retention deletes cheap on append-only data.

All timestamps are timezone-aware UTC. Retention: measurements, status events and cleared alerts older than
`TELEMETRY_RETENTION_DAYS` (default 30) are purged hourly (`python -m app.telemetry.db purge` does it on demand).
No extra time-series database is used.

The schema is versioned with Alembic (`app/telemetry/migrations`). Migrations run at startup;
`python -m app.telemetry.db upgrade` / `current` work from `backend/`. A test checks that the migration produces exactly
the model schema.

### Metric catalogue

`GET /api/telemetry/metrics` lists every metric with its canonical unit, plausibility range and state encoding
(`app/telemetry/catalog.py`).

| Equipment | Metrics |
|---|---|
| Battery | `battery_voltage` V, `battery_current` A (+ charge / − discharge), `battery_temperature` °C, `state_of_charge` %, `state_of_health` %, `battery_state` (−1 discharging, 0 idle, 1 charging) |
| Power supply unit | `input_voltage` V (AC), `output_voltage` V (DC), `output_current` A, `power_consumption` W, `temperature` °C, `operational_status` |
| Equipment cabinet | `internal_temperature` °C, `humidity` %, `door_open`, `smoke_detected`, `fan_status` |
| Router, radio unit | `temperature` °C, `power_consumption` W, `supply_voltage` V, `operational_status`, `network_available`, radio units also `rssi` dBm, `vswr` |
| Site | `ambient_temperature` °C, `ambient_humidity` %, `mains_available`, `backup_power_active`, `power_consumption` W |

Accepted unit spellings are converted to the canonical unit: mV/kV → V, mA → A, kW → W, °F/K → °C, `ratio` (0–1) → %.
An incompatible unit (amperes for a voltage) makes the reading `invalid`.

## Telemetry message format

The same JSON for HTTP and MQTT:

```json
{
  "sensor_id": "VOLT-001",
  "message_id": "VOLT-001-000123",
  "timestamp": "2026-11-07T10:15:00+05:30",
  "readings": [
    {"metric": "battery_voltage", "value": 52.4, "unit": "V"},
    {"metric": "battery_temperature", "value": null}
  ]
}
```

- `timestamp` must carry a timezone offset. If it is omitted, the server's receive time is used.
- `message_id` is optional but recommended. A repeat of it is ignored, which makes retries safe.
- `value: null` means the sensor reports the value as unavailable. It is stored as `missing` and never shown as a measurement.
- Up to 100 readings per message, and up to 500 messages per `POST /api/telemetry/ingest/batch`.

## Ingestion rules

| Check | Outcome |
|---|---|
| Sensor not registered | message **rejected** and logged; nothing is created or changed |
| Sensor disabled | rejected |
| Wrong channel (a physical sensor's data through the simulator, a simulated sensor's data through HTTP/MQTT, or HTTP data for an MQTT sensor) | rejected |
| Repeated `message_id` (per sensor) | ignored as duplicate |
| Same sensor + metric + timestamp as an earlier reading | ignored as duplicate (also enforced by the unique index) |
| Timestamp more than 60 s in the future | rejected (sensor clock problem) |
| Metric not configured for the sensor | that reading rejected |
| `value: null` | stored, quality `missing` |
| Unknown or incompatible unit, or physically implausible value (e.g. 400 V on a battery) | stored with `value` NULL, quality `invalid`, raw value kept |
| Older than the sensor's latest reading of that metric | stored, quality `out_of_order`: visible in history, **never** the current value, **never** raises an alert |

Batches are written in one transaction. Ingestion statistics and the last 50 rejections are in
`GET /api/telemetry/status`.

## Connection status and stale data

Each sensor has an expected sampling interval. Defaults: **stale** after 3 × interval, **offline** after
10 × interval (minimum 60 s). Both are configurable per sensor.

| Status | Meaning |
|---|---|
| `unknown` | registered but never reported |
| `online` | reported within `stale_after_s` |
| `stale` | late, but not yet offline |
| `offline` | silent for longer than `offline_after_s` |
| `disabled` | switched off in the registry |

A stale or offline sensor's readings are still returned, but with `current: false`, their timestamp and age. The
dashboard shows them as "last known · 5m ago" in a dimmed style, never as current values. Status changes are stored
in `sensor_status_events`; they feed the availability chart.

## Alerts

Rules live in `backend/configs/telemetry/thresholds.yaml` (no thresholds are hard-coded):

```yaml
- id: battery_temperature_high
  metric: battery_temperature
  above: {warning: 40, critical: 50}
  hysteresis: 2      # clears only below 38
  for_s: 60          # must persist 60 s before opening
  categories: [battery]   # optional scope
```

- One active alert per rule and sensor. It escalates or de-escalates between warning and critical, and records the
  peak value. It clears automatically (with hysteresis) when the value is back to normal.
- Acknowledgement (`POST /api/telemetry/alerts/{id}/ack` with your name and a note) is separate from clearing.
- **Communication alerts:** a sensor going offline opens a warning, which escalates to critical after 15 min and
  clears when the sensor reports again. Simulated sensors are exempt while the simulator is intentionally stopped or
  writing back-dated history.
- Only good, in-order readings are evaluated.
- Messages read "Detected condition: … Requires investigation." A threshold crossing is reported as a condition to
  check, never as a statement that the component is faulty.
- The default levels are generic starting points for a −48 V DC site. **Set them from the actual equipment's
  datasheets.**

## Registering equipment and sensors

Dashboard: **Equipment Monitoring → + Equipment**, then **Sensor Management → + Sensor**. The sensor table's
equipment column changes the association.

API:

```powershell
$h = @{ "Content-Type" = "application/json" }
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/telemetry/equipment -Headers $h -Body '{"equipment_id":"BAT-001","category":"battery","name":"Battery string 1","zone":"Cabinet A"}'
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/telemetry/sensors -Headers $h -Body '{"sensor_id":"VOLT-001","name":"String voltage","sensor_type":"voltage","protocol":"mqtt","metrics":["battery_voltage"],"equipment_id":"BAT-001","sampling_interval_s":10}'
Invoke-RestMethod -Method Put http://127.0.0.1:8000/api/telemetry/sensors/VOLT-001/equipment -Headers $h -Body '{"equipment_id":"BAT-002"}'
```

Rules:
- Several sensors may belong to one equipment item.
- A sensor's category follows its equipment.
- Site-level sensors have no equipment and use the category `site`.
- Re-associating a sensor does not rewrite history: every measurement keeps the equipment it was recorded for.
- Physical and simulated records cannot be mixed. Simulated equipment and sensors (`SIM-` IDs) are created only by
  the simulator; the API refuses them.
- Sensor `config` may not contain secrets. A key such as `password` is rejected; reference an environment variable
  instead (`password_env: MQTT_PASSWORD`).

## MQTT integration

1. Install a broker. For example, Mosquitto from mosquitto.org runs as a Windows service on port 1883.
2. In `backend/.env`:
   ```
   MQTT_ENABLED=true
   MQTT_HOST=127.0.0.1
   MQTT_PORT=1883
   MQTT_USERNAME=dt_backend   # optional
   MQTT_PASSWORD=...          # optional; only in .env, never logged or returned by the API
   MQTT_TLS=false
   ```
3. Register each sensor with `"protocol": "mqtt"`.
4. Devices publish the JSON message above (QoS 1) to `bts/<SITE_ID>/<sensor_id>/telemetry`. The `sensor_id` may be
   omitted from the payload; it is then taken from the topic. If both are present, they must match.

Test without hardware:

```powershell
mosquitto_pub -h 127.0.0.1 -t bts/SITE-001/VOLT-001/telemetry -q 1 -m '{"readings":[{"metric":"battery_voltage","value":52.3}]}'
```

The adapter reconnects with exponential backoff (1–60 s). Messages for another site, malformed JSON, a topic that
doesn't match the payload, or any other topic are dropped and counted (`GET /api/telemetry/status` → `mqtt`).

## Simulator

Configuration: `backend/configs/telemetry/simulator.yaml`. It defines one cabinet, a rectifier shelf, a 48 V battery
string with four sensors, a router, two radio units, and site environment and power sensors (12 sensors, all `SIM-`).

The values come from a physical model, not free random numbers:
- a daily ambient cycle (with optional heat wave);
- traffic-driven radio and router power;
- a −48 V DC bus fed by the rectifier;
- a 24-cell lead-acid battery whose state of charge integrates the charge or discharge current, with voltage from
  float voltage or open-circuit voltage minus I·R;
- cabinet temperature lagging ambient plus equipment heat.

Random mains outages, door openings and sensor dropouts follow the configured rates. If the battery runs flat
during a long outage, the radio units shut down.

```powershell
# from the dashboard: Sensor Management → Simulated sensor environment → Start
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/telemetry/simulator/start -ContentType application/json -Body '{"interval_scale":1,"backfill_hours":24}'
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/telemetry/simulator/scenario -ContentType application/json -Body '{"scenario":"mains_outage","duration_s":600}'
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/telemetry/simulator/stop
```

- `interval_scale` divides every sampling interval (up to 100×).
- `backfill_hours` (up to 7 days) first writes clearly simulated history at a coarse step, so the analytics charts
  have data. On this machine, 24 h takes about 55 s (about 35k messages).
- Scenarios: `mains_outage`, `sensor_dropout` (optionally `sensor_id`), `door_open`, `fan_failure`, `heat_wave`.
- The simulator never starts by itself. `SIMULATOR_AUTOSTART=true` starts it with the app, but never when
  `APP_ENV=production`; there, `start` is refused.
- It runs in one daemon thread and stops on app shutdown.

## API

All under `/api/telemetry` (OpenAPI at `/docs`). List endpoints return `{total, limit, offset, items}`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/status` | database kind, schema revision, MQTT and simulator state, ingest statistics, recent rejections (never secrets) |
| GET | `/metrics` | metric catalogue |
| GET | `/summary` | site summary: sensor counts by status and source, monitored equipment, site temperature, site power, alerts, readings/min |
| GET, POST | `/equipment` | list (filter `category`, `data_source`) / register physical equipment |
| GET, PATCH, DELETE | `/equipment/{id}` | read / edit / delete (sensors are detached, not deleted) |
| GET | `/equipment/{id}/telemetry` | equipment, its sensors, latest readings, active alerts |
| POST, DELETE | `/equipment/{id}/visual-links[/{link_id}]` | explicit link to a visual observation |
| GET, POST | `/sensors` | list (filters `sensor_type`, `category`, `equipment_id`, `protocol`, `data_source`, `status`; pagination) / register |
| GET, PATCH, DELETE | `/sensors/{id}` | read / edit configuration / delete |
| PUT | `/sensors/{id}/equipment` | associate (`{"equipment_id": "BAT-001"}`) or detach (`null`) |
| GET | `/sensors/{id}/status-history?range=24h` | connection-state changes |
| POST | `/ingest`, `/ingest/batch` | HTTP ingestion; header `X-API-Key` (403 if no key is configured, 401 if wrong) |
| GET | `/latest` | latest reading per sensor and metric (filters `sensor_id`, `equipment_id`, `metric`, `category`) |
| GET | `/history?metric=&range=15m\|1h\|6h\|24h\|7d` | bucketed avg/min/max per sensor (or `start`/`end`; `sensor_id`, `equipment_id`, `max_points`); empty when there is no data |
| GET | `/measurements` | raw measurements, paginated, filter by quality, source, time |
| GET | `/availability?range=24h` | share of time each sensor was online |
| GET | `/alerts` | history (filters `state`, `severity`, `equipment_id`, `sensor_id`, `acknowledged`; pagination) |
| POST | `/alerts/{id}/ack` | `{"acknowledged_by": "...", "note": "..."}` |
| GET | `/alerts/rules` | loaded threshold rules |
| GET | `/simulator` | simulator status |
| POST | `/simulator/start`, `/simulator/stop`, `/simulator/scenario` | simulator control |
| WS | `/ws/telemetry` | snapshot on connect, then `batch` messages with `measurements`, `sensor_status` and `alert` events (at most 4 per second) |

No existing detection, tracking, inventory or model endpoint changed.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `APP_ENV` | `development` | `production` disables the simulator |
| `TELEMETRY_ENABLED` | `true` | |
| `TELEMETRY_DATABASE_URL` | unset | `postgresql+psycopg://user:pass@host:5432/db`; unset = SQLite file `runtime/telemetry.db` (development fallback, logged as a warning) |
| `TELEMETRY_INGEST_API_KEY` | unset | required for HTTP ingestion; unset = ingestion closed |
| `TELEMETRY_RETENTION_DAYS` | 30 | |
| `TELEMETRY_THRESHOLDS_FILE` | `configs/telemetry/thresholds.yaml` | |
| `TELEMETRY_WATCHDOG_INTERVAL_S` | 2 | connection-state check period |
| `SITE_ID`, `SITE_NAME` | `SITE-001`, `BTS site` | |
| `SIMULATOR_CONFIG_FILE`, `SIMULATOR_AUTOSTART` | `configs/telemetry/simulator.yaml`, `false` | |
| `MQTT_ENABLED`, `MQTT_HOST`, `MQTT_PORT`, `MQTT_USERNAME`, `MQTT_PASSWORD`, `MQTT_TLS`, `MQTT_TOPIC`, `MQTT_CLIENT_ID`, `MQTT_QOS` | off, 127.0.0.1, 1883, –, –, false, `bts/+/+/telemetry`, `dynamic-twin-backend`, 1 | |

Secrets (`TELEMETRY_DATABASE_URL`, `TELEMETRY_INGEST_API_KEY`, `MQTT_PASSWORD`) are `SecretStr`: they are not logged
and are not returned by `/api/config` or `/api/telemetry/status`. Keep them in `backend/.env`, which git ignores.

## Database setup (PostgreSQL)

PostgreSQL 18 is installed on the development machine as the Windows service `postgresql-x64-18`. In an
administrator `psql`:

```sql
CREATE ROLE dt_app LOGIN PASSWORD '<choose one>';
CREATE DATABASE dynamic_twin OWNER dt_app;
```

Then in `backend/.env`:

```
TELEMETRY_DATABASE_URL=postgresql+psycopg://dt_app:<password>@127.0.0.1:5432/dynamic_twin
```

Run `.venv\Scripts\python.exe -m app.telemetry.db upgrade`, or start the API, which migrates automatically. If the
server is unreachable, the API logs the reason after a 5 s connection timeout and runs without telemetry.

## Testing

```powershell
cd backend
.venv\Scripts\python.exe -m pytest tests/test_telemetry.py -q     # telemetry only (~30 s)
.venv\Scripts\python.exe -m pytest -q                             # everything
```

`tests/test_telemetry.py` (33 tests) covers:
- migration and model parity;
- unit normalisation and plausibility;
- schema validation, including the secret rejection;
- registration and association, including history keeping the old equipment;
- several sensors per equipment;
- rejection of unregistered sensors and wrong channels;
- missing and invalid values;
- duplicates by message ID, by timestamp and by database constraint;
- out-of-order readings;
- future timestamps;
- the stale → offline → recovery lifecycle with communication alerts and availability;
- the threshold lifecycle (for_s, escalation, hysteresis), rule scoping, rule validation;
- persistence across a restart;
- history bucketing and empty series;
- the MQTT adapter with a mocked client;
- simulator physics bounds and the outage scenario, simulator lifecycle, dropouts and isolation, the production refusal;
- the REST API (auth, validation, pagination, acknowledgement);
- WebSocket snapshot and live updates;
- the unreachable-database fallback.

All tests use temporary SQLite files and need no broker, server or hardware.

## What works today and what needs hardware

| Capability | With the simulator | Needs real hardware or infrastructure |
|---|---|---|
| Registry, equipment association, storage, history, analytics, alerts, WebSocket dashboard | ✅ demonstrated | — |
| HTTP ingestion | ✅ tested with HTTP clients | a device or gateway that can POST JSON |
| MQTT ingestion | ✅ tested with a mocked client | an MQTT broker (Mosquitto) and devices or a gateway publishing to it |
| PostgreSQL storage | ✅ code and migration; tests run on SQLite | the `dynamic_twin` database and role (see above) |
| Battery, PSU, cabinet, router, radio, site measurements | simulated values only | sensors, BMS, rectifier controllers, SNMP or Modbus gateways |
| Modbus | interface only | device register maps (address, type, scale per metric) |
| Accurate thresholds | generic defaults | the installed equipment's datasheets |

## Adding a protocol

Implement `TelemetryAdapter` (`start`, `stop`, `status`) in `app/telemetry/adapters.py`, convert each incoming
message into `IngestMessage`, and call `pipeline.ingest(message, "<protocol>")`. Add the protocol name to
`catalog.PROTOCOLS`, and start the adapter in `TelemetryService.start`. Validation, storage, alerts and live updates
need no changes.

For Modbus specifically, each device needs its transport (TCP host/port or serial line), unit ID, and a register
map: address, function code, data type, scale, metric and unit.
