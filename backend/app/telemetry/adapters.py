"""Communication adapters. Each adapter turns protocol traffic into IngestMessage objects and hands them to the
ingestion pipeline; the pipeline and everything after it is protocol-independent.

  HTTP       POST /api/telemetry/ingest[/batch] (see api/telemetry_routes.py), authenticated with X-API-Key
  MQTT       MqttAdapter below (paho-mqtt), topic bts/<site_id>/<sensor_id>/telemetry, JSON payload = IngestMessage
  simulator  app/telemetry/simulator.py
  Modbus     interface only (ModbusAdapter); needs device register maps before it can be implemented

Adding a protocol = implementing `TelemetryAdapter` and registering it in TelemetryService.
"""

import json
import logging
import threading
from datetime import UTC, datetime
from typing import Protocol

from pydantic import ValidationError

from app.telemetry.schemas import IngestMessage

logger = logging.getLogger(__name__)


class TelemetryAdapter(Protocol):
    name: str

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def status(self) -> dict: ...


def parse_topic(topic: str) -> tuple[str, str] | None:
    """bts/<site_id>/<sensor_id>/telemetry -> (site_id, sensor_id)."""
    parts = topic.split("/")
    if len(parts) == 4 and parts[0] == "bts" and parts[3] == "telemetry" and parts[1] and parts[2]:
        return parts[1], parts[2]
    return None


class MqttAdapter:
    """Subscribes to telemetry topics and forwards valid messages. Network handling (reconnect with backoff)
    runs in paho's own thread; malformed messages are logged and counted, never raised."""

    name = "mqtt"

    def __init__(self, settings, pipeline, client_factory=None):
        self.s, self.pipeline = settings, pipeline
        self._factory = client_factory or self._paho_client
        self.client = None
        self.connected = False
        self.last_error: str | None = None
        self.counters = {"received": 0, "forwarded": 0, "malformed": 0, "wrong_site": 0, "topic_mismatch": 0}
        self._lock = threading.Lock()

    def _paho_client(self):
        import paho.mqtt.client as mqtt

        return mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id=self.s.mqtt_client_id)

    def start(self) -> None:
        c = self._factory()
        if self.s.mqtt_username:
            c.username_pw_set(self.s.mqtt_username,
                              self.s.mqtt_password.get_secret_value() if self.s.mqtt_password else None)
        if self.s.mqtt_tls:
            c.tls_set()
        c.reconnect_delay_set(min_delay=1, max_delay=60)
        c.on_connect = self.on_connect
        c.on_disconnect = self.on_disconnect
        c.on_message = self.on_message
        self.client = c
        try:
            c.connect_async(self.s.mqtt_host, self.s.mqtt_port, keepalive=30)
            c.loop_start()
            logger.info("MQTT adapter connecting to %s:%d (topic %s)", self.s.mqtt_host, self.s.mqtt_port,
                        self.s.mqtt_topic)
        except Exception as exc:  # noqa: BLE001 - the API must keep running without the broker
            self.last_error = f"connect failed: {type(exc).__name__}"
            logger.error("MQTT adapter could not start: %s", type(exc).__name__)

    def stop(self) -> None:
        if self.client is not None:
            try:
                self.client.disconnect()
                self.client.loop_stop()
            except Exception:  # noqa: BLE001
                pass
        self.connected = False

    def on_connect(self, client, userdata, flags, reason_code, properties=None):
        ok = getattr(reason_code, "is_failure", None) is False or reason_code == 0
        self.connected = bool(ok)
        if ok:
            client.subscribe(self.s.mqtt_topic, qos=self.s.mqtt_qos)
            self.last_error = None
            logger.info("MQTT connected; subscribed to %s", self.s.mqtt_topic)
        else:
            self.last_error = f"connection refused: {reason_code}"
            logger.error("MQTT connection refused: %s", reason_code)

    def on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self.connected = False
        if reason_code != 0:
            self.last_error = f"disconnected: {reason_code}"
            logger.warning("MQTT disconnected (%s); paho will reconnect with backoff", reason_code)

    def on_message(self, client, userdata, msg) -> None:
        with self._lock:
            self.counters["received"] += 1
        parsed = parse_topic(msg.topic)
        if parsed is None:
            self._count("malformed", f"unexpected topic {msg.topic!r}")
            return
        site_id, topic_sensor = parsed
        if site_id != self.s.site_id:
            self._count("wrong_site", f"message for site {site_id!r} ignored")
            return
        try:
            data = json.loads(msg.payload.decode("utf-8"))
            if isinstance(data, dict):
                data.setdefault("sensor_id", topic_sensor)
            message = IngestMessage.model_validate(data)
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
            self._count("malformed", f"invalid payload on {msg.topic}: {type(exc).__name__}")
            return
        if message.sensor_id != topic_sensor:
            self._count("topic_mismatch", f"payload sensor {message.sensor_id!r} != topic sensor {topic_sensor!r}")
            return
        self.pipeline.ingest(message, "mqtt", received_at=datetime.now(UTC))
        with self._lock:
            self.counters["forwarded"] += 1

    def _count(self, key: str, why: str) -> None:
        with self._lock:
            self.counters[key] += 1
        logger.warning("MQTT message dropped: %s", why)

    def status(self) -> dict:
        # never includes credentials
        return {"enabled": True, "connected": self.connected, "broker": f"{self.s.mqtt_host}:{self.s.mqtt_port}",
                "topic": self.s.mqtt_topic, "tls": self.s.mqtt_tls, "authenticated": bool(self.s.mqtt_username),
                "last_error": self.last_error, **self.counters}


class ModbusAdapter:
    """Interface placeholder for Modbus RTU/TCP. Implementing it needs, per device: transport (TCP host/port or serial
    line), unit id, and a register map (address, function code, data type, scale, metric, unit). Those come from the
    hardware's documentation, so nothing is assumed here."""

    name = "modbus"

    def __init__(self, *_, **__):
        raise NotImplementedError("Modbus is not implemented: it needs the devices' register maps (see "
                                  "docs/sensor-integration.md, 'Adding a protocol')")
