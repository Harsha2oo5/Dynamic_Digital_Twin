"""Anomaly detection, equipment health scoring and trend-based predictive maintenance (Milestone 3).

telemetry readings ──► features (1-min grid, rolling robust statistics) ──► detectors (robust z, IQR, rate of
change, Holt forecast residual, state transitions, Isolation Forest) ──► context (expected behaviour, e.g. battery
discharge during a mains outage) ──► anomaly episodes (grouped, de-duplicated) ──► health score ──► maintenance
recommendations (Theil-Sen trend + Mann-Kendall test).

Scores are threshold units, not probabilities. Nothing here predicts failure dates.
"""
