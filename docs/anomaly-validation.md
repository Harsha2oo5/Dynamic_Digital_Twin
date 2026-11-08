# Anomaly detection: simulation validation and CPU performance

> **SIMULATED DATA ONLY.** These runs show that the pipeline separates *simulator-injected* faults from *simulated*
> normal operation. The detectors were developed against this same simulator, so the figures are optimistic and are
> **not** real-world detection rates, false-alarm rates or failure-prediction accuracy. No physical telemetry or
> failure history exists yet.

Reproduce from `backend/`: `python -m app.analytics.validate --name <run> [--seed N] [--days 7]`. The full reports are
in `experiments/anomaly_validation/<run>/report.{md,json}`, which git ignores.

## Method

The harness (`app/analytics/validate.py`) builds a complete telemetry and analytics stack on a temporary SQLite
database. It drives the physical site model (`app/telemetry/simulator.py`) in simulated time, with one reading per
sensor per minute, through the **real ingestion pipeline, anomaly engine, health scoring and maintenance analysis**.
Random mains outages (about 1.5 per day, 10–45 min) and door openings (about 2 per day) stay switched on throughout,
because they are normal site life.

| Simulated time | Content |
|---|---|
| day 0–3 | normal operation; Isolation Forest models are trained on it at day 3 (outage and recharge periods excluded) |
| day 3–4 | normal operation with every detector and the trained models active |
| day 4 + 2 h | **temperature_spike**: radio unit 1 runs 18 °C hotter for 45 min (sudden, then recovery) |
| day 4 + 8 h | **power_anomaly**: radio unit 2 draws +250 W for 60 min |
| day 4 + 14 h | **voltage_fluctuation**: rectifier output oscillates ±0.8–1.4 V minute to minute for 30 min |
| day 4 + 18 h | **sensor_dropout**: radio unit 1's sensor is silent for 20 min (communication failure, not an equipment fault) |
| day 5 → 7 | **battery_degradation** (gradual over 48 h): string voltage −2.5 V, +6 °C, state of health −8 % |
| day 5.5 → 7 | **psu_overheating** (gradual over 36 h): rectifier temperature +15 °C |

**Scoring:**
- A scenario is *detected* if an anomaly on the expected equipment opens between its start and 15 min after its end.
- Gradual scenarios also pass with a matching trend recommendation.
- The dropout passes if a communication alert opens **and** no equipment anomaly opens.
- False alarms are anomalies opened anywhere else after the 2 h warm-up.

## Results (run `v4`, seed 7)

| Scenario | Detected | Delay | Max severity | Detectors | Ended after recovery | Trend recommendation |
|---|---|---|---|---|---|---|
| temperature_spike | ✅ | 2 min | critical | rate_of_change | ✅ | — |
| power_anomaly | ✅ | 2 min | critical | forecast_residual, rate_of_change, robust_z | ✅ | — |
| voltage_fluctuation | ✅ | 2 min | critical | forecast_residual, iqr, robust_z | ✅ | — |
| sensor_dropout | ✅ communication alert opened and cleared; **0 equipment anomalies** | | | | | |
| battery_degradation | ✅ via trends | — | — | — | — | battery voltage −0.39 V/day and battery temperature +0.56 °C/day, both **medium** evidence |
| psu_overheating | ✅ | 85 min | warning | isolation_forest | — | PSU temperature +0.99 °C/day, **high** evidence |

**Normal operation:**
- **2 false alarms in 7 simulated days** (≈ 0.6 per day across 6 equipment items plus site sensors), both warnings:
  - a PSU power drop after a recharge (robust z 12.3 against threshold 10);
  - one Isolation Forest event on an unusually dry, hot cabinet afternoon.
- No false alarm on day 3, when the trained models were active.
- **16 random mains outages**; 24,048 evaluations were explained by the outage/recharge context instead of being flagged.

**Severity separation:** sudden faults reached critical, gradual degradation produced warnings and trend
recommendations, and false alarms were warnings.

**Health at the end:**
- PSU **warning** (71): overheating trend plus recent anomalies.
- Battery **healthy** (90): its degradation shows as medium-evidence maintenance trends rather than anomalies.
- Everything else healthy (97–100).
- Communication problems did not lower any health score.

## How the results evolved (what was wrong and what changed)

| Run | Detected | False alarms / 7 days | Change that followed |
|---|---|---|---|
| v2 | 5/6 (voltage fluctuation missed) | 8 (1 critical) | Split the context: outage-only metrics (supply/output voltage) vs recharge metrics (charge current, PSU load, battery heat), with recharge detected from battery current instead of a fixed 3-h timer. Robust-z overrides for daily-cycle metrics (power 10, temperatures 8). Door transition limit raised to 10 |
| v3 | 5/6 (still missed) | 19 (16 critical, PSU output voltage) | Investigation: after an outage the Holt forecast carried an artificial trend from 0 V back to 53.5 V, overshooting for over an hour. That produced the false alarms and kept the episode open, which absorbed the fluctuation. Fix: reset the forecast during context periods, and treat "value closer to baseline than the forecast" as recovery |
| v4 | **6/6** | **2** (warnings) | — |

Earlier, unit tests had already led to two other changes:
- **Isolation Forest uses value-only features.** Rate-of-change features diluted the isolation of unusual combinations.
- **Rate of change scores 0 for moves back towards the baseline**, so recoveries aren't flagged.

Because these choices were tuned on seed 7, the same evaluation was repeated on an **unseen seed** (below).

## Unseen seed (run `v5_seed11`)

Same configuration and code, seed 11. This means different random outages (6 instead of 16), door openings and noise,
none of which were seen while tuning.

| Scenario | Result |
|---|---|
| temperature_spike | ✅ 2 min, critical |
| power_anomaly | ✅ 2 min, critical (robust z, rate of change, forecast, Isolation Forest) |
| voltage_fluctuation | ✅ 2 min, critical |
| sensor_dropout | ✅ communication alert opened and cleared, 0 equipment anomalies |
| battery_degradation | ✅ via the battery temperature trend (+0.54 °C/day, medium). The voltage trend did **not** reach significance with this seed |
| psu_overheating | ✅ high-evidence trend (+1.0 °C/day). The anomaly itself came only after 28 h (Isolation Forest, warning) |

**1 false alarm in 7 days** (Isolation Forest, warning), and none on the normal day 3.

Conclusion: the tuned configuration held on the unseen seed. Gradual degradation is reliably surfaced by the trend
analysis rather than by the anomaly detectors, which is by design: rolling baselines adapt to slow drift. Which trend
reaches significance varies with the random outage pattern.

## CPU performance

Measured on the development machine (Intel i7-1255U, 16 GB RAM, CPU only). The run was a 24 h simulator backfill
(34,560 messages, about 112k readings, 12 sensors, 29 analysed series), with nothing else running.

| Measurement | Result |
|---|---|
| Ingestion without analytics | 2,313 messages/s |
| Ingestion with the anomaly engine | 1,499 messages/s (≈ 35 % overhead, during a backfill burst) |
| Detector evaluation (features + univariate detectors), per closed bucket | 0.20 ms mean |
| Isolation Forest scoring, one vector (200 trees) | 9.5 ms |
| Health refresh, all 6 equipment items | 15 ms |
| Maintenance analysis (7-day trends, all equipment) | 373 ms |
| Training Isolation Forest for 5 categories on 24 h | 11.7 s |
| Full 7-day validation (120,940 messages, ~290k evaluations) | 12 min |

**Live load at this site's size:** 29 series evaluated once per minute (about 6 ms) plus 5 Isolation Forest
scorings per minute (about 47 ms). That is well under 0.1 % of one CPU core. The 1.2 ms mean per evaluation in the
validation reports includes Isolation Forest calls. Isolation Forest is the most expensive step per call
(scikit-learn's per-call overhead); it runs at most once per minute per equipment item.
