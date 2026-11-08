# Anomaly detection, equipment health and predictive maintenance (Milestone 3)

This milestone adds a predictive-intelligence layer on top of the telemetry from Milestone 2. It does four things:
- continuously scores equipment telemetry for abnormal behaviour;
- keeps a history of anomaly records that operators can acknowledge and resolve;
- maintains an explainable health index per equipment item;
- flags degradation trends that call for an inspection.

> **What these numbers are, and are not**
> - Anomaly **scores** are distances from learned normal behaviour, in *threshold units* (1.0 = at the detector's
>   threshold). They are **not probabilities**.
> - The **health score** is an index of recent abnormal behaviour. It is **not a failure probability**.
> - Maintenance recommendations are **trend-based indications**. With no historical failure data, **no failure
>   dates** are predicted.
> - All validation so far uses **simulated** data, and says nothing about real-world accuracy.

## Contents
1. [Architecture](#architecture)
2. [Features](#preprocessing-and-features)
3. [Algorithms](#algorithms-and-why)
4. [Equipment-specific detection](#equipment-specific-detection)
5. [Real-time operation](#real-time-operation)
6. [Health score](#health-score)
7. [Predictive maintenance](#predictive-maintenance)
8. [Data model and API](#data-model-and-api)
9. [Models and configuration](#model-management-and-configuration)
10. [Dashboard](#dashboard)
11. [Simulation validation](#simulation-validation)
12. [Performance](#cpu-performance)
13. [Testing](#testing)
14. [Limitations](#limitations)

## Architecture

```
telemetry ingestion (validated, normalised, de-duplicated, ordered; Milestone 2)
   │ every batch of new in-order good readings (listener hook)
   ▼
AnalyticsEngine (backend/app/analytics/engine.py)
   ├─ SeriesState per (equipment, sensor, metric): 1-minute grid, gaps, rolling robust stats, Holt forecast
   ├─ univariate detectors per metric (robust z, IQR, rate of change, forecast residual, state transitions)
   ├─ Isolation Forest per category (trained offline; same data source only)
   ├─ context rules (mains outage / recharge = expected battery and PSU behaviour)
   └─ episodes ─► anomalies table ─► WS /ws/telemetry {"type":"anomaly"}
AnalyticsService worker (30 s): equipment health ─► WS {"type":"health"}; 5-min snapshots ─► equipment_health_history
                                maintenance analysis (15 min) ─► maintenance_recommendations ─► WS {"type":"recommendation"}
API /api/analytics/*  ·  training: python -m app.analytics.train  ·  validation: python -m app.analytics.validate
```

Code is in `backend/app/analytics/`. Configuration is in `backend/configs/analytics/anomaly.yaml` (versioned in the
database). The analytics tables are added by Alembic migration `0002`.

## Preprocessing and features

`features.py` runs the same code for training (replayed history) and real-time inference, so the two cannot drift.

| Concern | Handling |
|---|---|
| Irregular sampling | readings are averaged into 1-minute buckets of **measurement time** |
| Missing readings / sensor silent | empty buckets are gaps (NaN), never interpolated. A gap longer than 5 buckets resets the forecast and blocks rate-of-change across it, so reconnecting sensors don't look like sudden changes |
| Missing / invalid / out-of-order readings | never reach analytics: only `good`, in-order readings are passed on (Milestone 2 rules) |
| Duplicates | removed by ingestion (message ID and unique timestamp) |
| Late readings | ignored (their bucket is closed) |
| Outliers | **not deleted**. Buckets that were flagged anomalous are kept out of the *baseline*, so a persistent fault doesn't become "normal". If they make up more than half the window, the shift is accepted as the new normal |
| Normalisation | deviations are divided by a robust spread (1.4826 × MAD), with a per-metric minimum (`scale_floor`) so flat signals can't produce huge scores |
| Identity | every feature and anomaly keeps equipment ID, sensor ID and timestamps |

Features of each closed bucket, relative to the previous 6 hours:
- value;
- mean, standard deviation, min, max, median, MAD, Q1/Q3;
- percentage deviation from the median;
- 10-minute rate of change, and the history of 10-minute changes;
- Holt (level + trend) one-step forecast and residual, and the residual history;
- data availability;
- number of state transitions (for state metrics).

All windows are configurable (`grid_s`, `baseline_buckets`, `short_buckets`, `min_history_buckets`).

## Algorithms (and why)

| Detector | Catches | Notes |
|---|---|---|
| `robust_z` | level shifts: value far from its 6 h median | default threshold 6 robust SD |
| `iqr` | values outside the Q1/Q3 fences (k = 3) | used for PSU output voltage |
| `rate_of_change` | sudden jumps or drops: 10-min change vs the baseline's 10-min changes | **returns to baseline score 0** (recovery isn't an anomaly). This was added after tests showed the drop back to normal being flagged for 10 minutes |
| `forecast_residual` | spikes and oscillation: Holt one-step forecast error vs past errors | voltage fluctuation, power spikes. The forecast is reset during context periods, and a value closer to its baseline than the overshooting forecast scores 0. Both were added after validation run v3 (see the validation doc) |
| `transitions` | flapping state: door openings, network or fan toggling | state metrics only |
| `isolation_forest` | unusual **combinations** of in-range values (e.g. high radio temperature at low traffic power) | per category, trained offline |

Choices:
- **Isolation Forest features are values only.** Adding 10-minute changes diluted the isolation: unusual radio-unit
  temperature/power pairs scored 0.3–0.4 threshold units instead of 1.3–1.6. Sudden changes are already covered by
  `rate_of_change`.
- **Isolation Forest can't flag values beyond its training range better than the most extreme training points.** A
  value outside the range ends up in the same extreme leaf. That case belongs to `robust_z`; the two are complementary.
- **One-Class SVM was not implemented.** Training cost grows roughly quadratically with samples (days of 1-minute data
  per category is tens of thousands of points on CPU). Its result depends heavily on the RBF kernel width and `nu`,
  with no labelled data to tune them. It would duplicate Isolation Forest's role (unusual combinations) without a
  measurable benefit here. The detector interface allows adding it later.
- **Sequence models (LSTM, autoencoders) were not used.** There is no real telemetry history to learn from yet, and
  they would add heavy CPU dependencies. Holt forecasting plus robust statistics cover sudden changes; Theil–Sen
  trends cover gradual degradation.

## Equipment-specific detection

Per category, `anomaly.yaml` lists the monitored metrics, the detectors for each, a multivariate feature set, and the
metrics that are *expected* to change during a mains outage.

| Category | Metrics → detectors | Context |
|---|---|---|
| Battery | voltage (z, rate, forecast) · temperature (z, rate) · current (z, forecast) · SOC (rate) · IF on voltage+current+temperature | all four metrics are `recharge_expected`: not flagged during an outage, while the battery recharges afterwards (charge current > 2 A), or for 1 h after charging ends (at least 30 min, at most 6 h after mains returns) |
| Power supply unit | output voltage (z, forecast, IQR) · output current · power (z, rate) · temperature (z, rate) · input voltage · IF | input/output voltage: `mains_outage_expected` (only during the outage). Current, power, temperature: `recharge_expected` |
| Equipment cabinet | internal temperature, humidity (z, rate) · door and fan state (transitions) · IF | |
| Router, radio unit | temperature (z, rate) · power (z, rate, forecast) · supply voltage · VSWR, RSSI (radio) · network state (transitions) · IF on temperature+power | supply voltage during an outage |
| Antenna, cable | VSWR (z, rate) **only if such sensors exist** | nothing is evaluated without readings |
| Site | ambient temperature, humidity | |

Threshold alerts from Milestone 2 stay as they are. They answer a different question ("is a limit crossed?") and
feed the health score.

## Real-time operation

1. A new reading is validated and stored by the telemetry pipeline, which then calls the analytics listener.
2. The engine looks up sensor, equipment and category context.
3. It updates the series' open 1-minute bucket. Nothing else happens until that bucket closes, so the work is at most
   once per minute per series, not per reading.
4. When the bucket closes, the configured detectors run (after a 1 h warm-up per series).
5. Context rules turn expected behaviour into score 0, and restart the forecast once that behaviour is over.
   Metrics with a strong daily cycle use higher robust-z thresholds (`detector_overrides`: power 10, temperatures 8).
6. Episodes: an anomaly opens after 2 consecutive evaluations ≥ 1.0. Severity is warning, or critical at ≥ 2.0.
   Detectors firing on the same equipment metric are grouped into one record. A recurrence within 30 min re-opens the
   same record (`occurrences` + 1). The condition ends after 5 evaluations below 0.7 and is then auto-resolved by
   `system`.
7. Records are stored in `anomalies` and pushed to the dashboard.

Each series keeps independent state. Several sensors on one equipment item are separate series under the same
equipment.

**Sensor communication is separate from equipment health.** Gaps are never evaluated, so a disconnected sensor can't
open, end or extend an equipment anomaly. Communication failures stay Milestone 2 communication alerts. They lower the
**data-quality** figure, not the health score. A simulator that's stopped on purpose produces no communication alerts.

After a restart, open episodes are restored from the database (no duplicates), and feature windows are re-warmed from
the last 6 hours of stored telemetry.

## Health score

`backend/app/analytics/health.py`, recomputed every 30 s:

```
score = clamp(100 − Σ penalties, 0, 100)

anomaly (last 24 h)  base[warning 12 | critical 30] × 2^(−hours since last active / 6)
                     + 4 × min(hours active, 3) + 3 × (occurrences − 1)          capped at 45 per anomaly
threshold alert      warning 8 | critical 20 while active   (communication alerts excluded)
maintenance trend    medium 5 | high 10 while active

healthy ≥ 85 · warning ≥ 65 · degraded ≥ 40 · critical below
```

- **Unknown:** no score is given when the equipment has no sensors for its configured metrics, or when fewer than 50 %
  of those metrics have fresh (sensor online) and warmed-up data. The reason is shown.
- **Data quality:** the average share of the last 24 h that the equipment's sensors were online, shown separately.
- Every score comes with its penalty breakdown. Snapshots are stored every 5 minutes for the trend chart.
- All weights are configurable.

## Predictive maintenance

`backend/app/analytics/maintenance.py`, every 15 minutes or via `POST /api/analytics/maintenance/run`:

1. **Hourly medians** of good readings over 7 days. For mains-dependent metrics (battery voltage, PSU output), hours
   during an outage and the 3 h recharge afterwards are excluded.
2. **Theil–Sen slope** (robust to outliers) with its 95 % confidence interval, plus **Kendall's tau** test for a
   monotonic trend.
3. **A degradation trend is reported when all hold:** direction as configured, slope ≥ `min_slope_per_day` (e.g.
   battery temperature +0.5 °C/day), confidence interval excludes zero, p < 0.05, and at least 48 h and 36 hourly
   points.
4. **Evidence level:** high (p < 0.001, ≥ 5 days, ≥ 96 points), medium (p < 0.01, ≥ 3 days), otherwise low. Hourly
   medians are autocorrelated, so p-values are optimistic; the slope and span requirements guard against that.
5. **Anomaly frequency:** at least 3 anomalies in the last 24 h, and twice the earlier daily average.

Each recommendation stores:
- equipment ID and category;
- the detected condition;
- supporting telemetry (the hourly series);
- related anomaly IDs;
- an inspection action from the config;
- evidence level and statistics;
- timestamps.

It resolves itself when a later analysis with enough data no longer finds the trend.

## Data model and API

Migration `0002` adds:

| Table | Purpose |
|---|---|
| `anomalies` | episodes: equipment, sensor, category, metric, kind, detector(s), model, observed, expected, unit, score, peak score, severity, status (open, acknowledged, resolved), condition active, first/last detected, ended, occurrences, explanation, evidence (features and every finding), data source, acknowledge/resolve audit |
| `equipment_health_history` | 5-minute health snapshots with components |
| `maintenance_recommendations` | condition, action, evidence level and statistics, status, acknowledgement |
| `anomaly_models` | algorithm, version, data source, artifact, config hash, features, parameters, thresholds, training window, evaluation |
| `anomaly_configs` | versioned YAML configuration; exactly one active |

Indexes cover equipment+time, status+severity, time, and the episode key.

All under `/api/analytics`. Lists are paginated `{total, limit, offset, items}`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/status` | availability, config version, loaded models, engine statistics |
| GET | `/health/summary` | counts per status, mean score, active and open anomalies, equipment requiring inspection |
| GET | `/health/equipment` · `/health/equipment/{id}` · `/health/equipment/{id}/history?range=` | health with components; snapshots |
| GET | `/anomalies` | filters: equipment, sensor, category, severity, status, metric, detector, active, data source, start/end |
| GET | `/anomalies/{id}` | details |
| POST | `/anomalies/{id}/ack` · `/anomalies/{id}/resolve` | `{"by": "...", "note": "..."}`; resolving an already resolved anomaly is a 409 |
| GET | `/recommendations` · POST `/recommendations/{id}/ack` | maintenance |
| POST | `/maintenance/run` | run the trend analysis now |
| GET | `/equipment/{id}/details` | identity, visual links, sensors, latest readings, health and its history, anomalies, recommendations, threshold alerts |
| GET, PUT | `/config` · GET `/config/versions` | YAML configuration (validated; 422 on errors) |
| GET | `/models` · POST `/models/{id}/activate` · POST `/models/reload` | anomaly models |

Live events on `/ws/telemetry`: `anomaly` (opened, escalated, updated, ended, reopened, acknowledged, resolved),
`health` and `recommendation`. No earlier endpoint or message changed.

## Model management and configuration

- **Training is offline and separate:**
  `python -m app.analytics.train --data-source simulated --days 7`.
  It replays stored telemetry through the live feature code, excludes outage and recharge periods, holds out the
  latest 20 %, and records:
  - algorithm, version, data source and training window;
  - features, parameters and thresholds (0.999 training quantile);
  - the hold-out flagged rate.

  Artifacts go to `backend/models/anomaly/` (not in git). Activate with the API or `--no-activate` / `activate`, then
  `POST /api/analytics/models/reload`.
- **Data source isolation:** a model only scores equipment of the data source it was trained on, so simulated-data
  models never touch physical equipment.
- **Configuration changes:** `PUT /api/analytics/config` with the complete YAML. It is validated (unknown metrics or
  detectors, inconsistent bands, a missing inspection action are all rejected), stored as a new version, applied
  immediately, and re-warmed from stored history. The anomaly model records the config hash it was trained with.
- **Reproducibility:** fixed seeds, a stored config hash, the training window in model metadata, and a deterministic
  validation harness.

## Dashboard

| Page | Content |
|---|---|
| Overview | equipment-health block: overall mean, healthy/warning/degraded/critical/unknown counts, active anomalies, recent anomaly activity, equipment requiring inspection |
| Equipment Health | card per equipment: score and status, coverage, data quality, top penalties, 24 h health sparkline, the scoring method |
| Anomaly Monitoring | live feed; filterable, paginated records; expandable evidence (observed vs expected, every detector's explanation, context); acknowledge and resolve |
| Predictive Maintenance | recommendations with evidence level, statistics, supporting hourly series chart, inspection action, acknowledge; "Analyse now"; anomalies per day per equipment (7 days) |
| Equipment details (`#details/<id>`) | identity, health with breakdown and trend, linked visual observations, sensors and live readings, historical telemetry chart, anomalies, recommendations, threshold alerts |

Simulated records keep their SIMULATED or `sim` badge everywhere.

## Simulation validation

Simulated only; details and history in [anomaly-validation.md](anomaly-validation.md).

| | Seed 7 (tuned on) | Seed 11 (unseen) |
|---|---|---|
| Injected scenarios passed | 6/6 | 6/6 |
| Sudden faults (temperature spike, power, voltage fluctuation) | detected in 2 min, critical, ended after recovery | same |
| Sensor dropout | communication alert only, 0 equipment anomalies | same |
| Gradual degradation | medium/high-evidence trend recommendations | same (battery: temperature trend only) |
| False alarms in 7 days | 2 (warnings) | 1 (warning) |

## CPU performance

Details in [anomaly-validation.md](anomaly-validation.md#cpu-performance):
- 0.2 ms per detector evaluation;
- 9.5 ms per Isolation Forest scoring;
- health refresh 15 ms;
- maintenance analysis 373 ms;
- about 35 % ingestion overhead during bursts.

Live load at this site is under 0.1 % of a core.

## Testing

`tests/test_analytics.py` has 40 tests and runs in about 20 s:
- feature grid and gaps;
- rate-of-change gap blocking and forecast reset;
- baseline exclusion and relearning;
- the scale floor;
- insufficient-data handling in every detector;
- IQR, rate of change, forecast residual, transitions;
- config validation (7 cases);
- normal operation not flagged;
- injected step detected (critical), then ended and auto-resolved;
- a moderate ramp stays a warning;
- single blips ignored;
- recurrence re-opens instead of duplicating;
- sensor gaps aren't anomalies;
- mains-outage context;
- Isolation Forest training, scoring and data-source isolation;
- health bands, decay, communication alerts excluded, unknown on insufficient data;
- trend detection, evidence levels, outage-hour exclusion, anomaly frequency;
- API lifecycle (anomaly → health → ack → resolve → 409/404 → details);
- unknown health via the API;
- config versioning and validation;
- WebSocket anomaly events;
- persistence across a restart;
- simulator fault scenarios;
- outage-only vs recharge context, and per-metric detector overrides;
- regressions from the validation runs: no forecast false alarm when voltage returns after an outage, and an
  oscillation shortly after an outage is still detected.

## Limitations

- **No real telemetry yet.** Every baseline, model and validation number comes from the simulator, whose physics the
  detectors are partly tuned to. Real sensors will need a warm-up period and threshold review.
- Thresholds, penalties and slopes are reasoned defaults, not fitted to labelled real anomalies (none exist).
- Isolation Forest models are trained on simulated data, so they only ever apply to simulated equipment until
  physical history exists and models are trained on it.
- Baselines are 6-hour rolling windows. Daily cycles push normal scores up to about 0.8 at the steepest part of the
  cycle; slower seasonal effects are not modelled.
- Trend p-values are optimistic because of autocorrelation.
- Health is an index, and its weights are configurable judgement calls.
