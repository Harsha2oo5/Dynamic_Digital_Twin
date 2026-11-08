"""Sensor integration and telemetry monitoring.

Sensor devices -> adapters (HTTP, MQTT, simulator; Modbus interface only) -> ingestion pipeline (validate, normalise,
registry check, de-duplicate, order) -> storage (PostgreSQL, SQLite fallback for development) -> monitoring engine
(latest values, staleness, threshold alerts) -> REST API + WebSocket -> dashboard.

Nothing here assumes physical hardware exists. Every measurement carries `data_source` ("physical" or "simulated"),
and simulated devices are registered as such.
"""
